import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from layers import GraphBuilder
from utils import Constants
# from torch.autograd import Variable # Variable is deprecated, direct tensor usage is preferred
from layers.Commons import DynamicGraphNN, GraphNN, Fusion, TimeAttention
from layers.TransformerBlock import TransformerBlock

from helpers.BaseLoader import BaseLoader
from helpers.BaseRunner import BaseRunner

class DyHGCN(nn.Module):
    """
    DyHGCN-S model.

    Core modifications and fixes:
    - Fixed temporal data leakage: replaced block-level time indexing with per-position causal indexing and future-time masking in time attention.
    - Added explicit time padding `Constants.PAD_TIME` and updated data loading to use it for timestamps.
    - Ensured device consistency and removed deprecated `.cuda()` usage patterns.
    - Improved numerical stability for time attention when all time keys are masked.

    The original implementation leaked future diffusion embeddings; this version enforces strict temporal causality.
    """
    Loader = BaseLoader
    Runner = BaseRunner

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--pos_dim', type=int, default=8,
                            help='Dimension for positional encoding, in the original implement is 8.')
        parser.add_argument('--n_heads', type=int, default=8,
                            help='Number of the attention head, in the original implement is 8.')
        parser.add_argument('--time_step_split', type=int, default=8,
                            help='Number of windows size.')
        return parser

    def __init__(self, args, data_loader):
        super(DyHGCN, self).__init__()
        self.device = args.device  # ADDED: Store the device from args
        self.user_num = data_loader.user_num
        self.embedding_size = args.d_model
        self.pos_dim = args.pos_dim
        self.n_heads = args.n_heads
        self.time_step_split = args.time_step_split
        self.dropout = nn.Dropout(args.dropout)
        self.drop_timestamp = nn.Dropout(
            args.dropout)  # Note: This is defined but not used in the provided forward pass

        # In the original paper, the dropout is 0.5
        self.gnn_layer = GraphNN(self.user_num, self.embedding_size, dropout=0.5)
        self.gnn_diffusion_layer = DynamicGraphNN(self.user_num, self.embedding_size, self.time_step_split)
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)

        self.time_attention = TimeAttention(self.time_step_split, self.embedding_size)
        self.decoder_attention = TransformerBlock(input_size=self.embedding_size + self.pos_dim, n_heads=self.n_heads)
        self.linear = nn.Linear(self.embedding_size + self.pos_dim, self.user_num)

        # self.relation_graph = GraphBuilder.build_friendship_network(data_loader)  # load friendship network
        # diffusion_graph might be a list of adjacency matrices (tensors) or other structures.
        # If they are tensors, they should be moved to the device.
        # Assuming build_dynamic_heterogeneous_graph returns data that gnn_diffusion_layer can handle.
        # If it returns tensors that need to be on the device directly, they should be moved here.
        # For now, we assume gnn_diffusion_layer handles device placement internally or its components are nn.Modules.
        self.diffusion_graph = GraphBuilder.build_dynamic_heterogeneous_graph(data_loader, self.time_step_split)
        # Example if diffusion_graph is a list of tensors:
        # self.diffusion_graph = [g.to(self.device) for g in GraphBuilder.build_dynamic_heterogeneous_graph(data_loader, self.time_step_split) if isinstance(g, torch.Tensor)]

        self.init_weights()
        # ADDED: Ensure all model parameters are on the correct device
        # This is usually handled by model.to(device) in the runner, but explicit here can be a safeguard
        # self.to(self.device) # Generally, this is done once on the top-level model instance

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def forward(self, input_seq, input_timestamp, tgt_idx):
        # input_seq, input_timestamp, tgt_idx are expected to be on self.device from the DataLoader/Runner
        # Truncate the last element; keep prior steps for next-step prediction
        input_seq = input_seq[:, :-1]  # [bth, seq_len]
        input_timestamp = input_timestamp[:, :-1]  # [bth, seq_len]

        # Create mask: padding positions (Constants.PAD) are True; others False
        mask = (input_seq == Constants.PAD)  # [bth, seq_len]

        # batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).cuda()  # [bth, seq_len]
        batch_t = torch.arange(input_seq.size(1), device=self.device).expand(
            input_seq.size())  # MODIFIED: Create on self.device
        # Apply dropout to positional embeddings to reduce overfitting
        order_embed = self.dropout(self.pos_embedding(batch_t))  # [bth, seq_len, pos_dim]

        # Get batch size and max sequence length
        batch_size, max_len = input_seq.size()

        # Initialize dyemb tensor to store dynamic node embeddings (batch_size, max_len, user_num)
        # dyemb = torch.zeros(batch_size, max_len, self.user_num).cuda()
        # dyemb is reassigned later by self.time_attention, so this initialization might not be strictly needed
        # If it were used directly before reassignment, it should be on device.
        # For now, commenting out as it seems unused before being overwritten.
        # dyemb_init_placeholder = torch.zeros(batch_size, max_len, self.user_num, device=self.device) # MODIFIED: Create on self.device

        # Define time step length for dynamic embedding updates
        step_len = 5

        # Compute each time step's dynamic node embedding graph
        dynamic_node_emb_dict = self.gnn_diffusion_layer(self.diffusion_graph)  # 8 time windows

        # Initialize dyemb_timestamp to store per-position index into dynamic embeddings
        dyemb_timestamp = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        # Get dynamic embedding dict's all time stamps, and create a mapping dict
        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_tensor = torch.tensor(dynamic_node_emb_dict_time, device=self.device, dtype=torch.float)

        # Compute causal time index per position to avoid future leakage from block-level indexing
        for pos in range(max_len):
            ts = input_timestamp[:, pos]  # [batch]
            # Mark valid timestamps (exclude PAD_TIME)
            valid_mask = ts > float(Constants.PAD_TIME)
            # Compare all time keys with current position timestamp, select not more than ts last key
            cmp = (dynamic_node_emb_dict_time_tensor.unsqueeze(0) <= ts.unsqueeze(1))  # [batch, time_step]
            idx = cmp.sum(dim=1) - 1  # Rightmost index <= ts
            idx = idx.clamp(min=0)    # Fall back to 0 when none are valid (ts too early or invalid)
            # Use 0 index for invalid time (does not peek future)
            idx = torch.where(valid_mask, idx, torch.zeros_like(idx))
            dyemb_timestamp[:, pos] = idx

        # Build list of user embeddings across time steps
        dyuser_emb_list = list()
        for val_key in sorted(dynamic_node_emb_dict.keys()):
            dyuser_emb_sub = F.embedding(input_seq, dynamic_node_emb_dict[val_key]).unsqueeze(2)
            dyuser_emb_list.append(dyuser_emb_sub)

        if not dyuser_emb_list:
            dyemb = torch.zeros(batch_size, max_len, self.embedding_size, device=self.device)
        else:
            dyuser_emb = torch.cat(dyuser_emb_list, dim=2)  # [bth, seq_len, time_step, hidden_size]

            # Construct time-attention mask to block future time windows
            time_keys = dynamic_node_emb_dict_time_tensor  # [time_step]
            time_mask = torch.zeros(batch_size, max_len, len(dynamic_node_emb_dict_time), dtype=torch.bool, device=self.device)
            for pos in range(max_len):
                ts = input_timestamp[:, pos]
                # Mask columns where time_keys > ts as True (future)
                future = (time_keys.unsqueeze(0) > ts.unsqueeze(1))
                # For invalid timestamps, mark entire row True to skip attention
                invalid = ts <= float(Constants.PAD_TIME)
                future = torch.where(invalid.unsqueeze(1), torch.ones_like(future), future)
                time_mask[:, pos, :] = future

            dyemb = self.time_attention(dyemb_timestamp, dyuser_emb, mask=time_mask)

        # Apply dropout to dynamic embeddings
        dyemb = self.dropout(dyemb)

        # Concatenate dynamic and positional embeddings along last dimension
        # final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()  # [bth, seq_len, hidden_size+pos_dim]
        final_embed = torch.cat([dyemb, order_embed],
                                dim=-1)  # [bth, seq_len, hidden_size+pos_dim] # MODIFIED: Removed .cuda()

        # Compute self-attention via decoder_attention and use mask to handle padding
        # att_out = self.decoder_attention(final_embed.cuda(),
        #                                  final_embed.cuda(),
        #                                  final_embed.cuda(),
        #                                  mask=mask.cuda())  # [batch_size, seq_len, hidden_size+pos_dim]
        # final_embed and mask are already on self.device
        att_out = self.decoder_attention(final_embed,
                                         final_embed,
                                         final_embed,
                                         mask=mask)  # [batch_size, seq_len, hidden_size+pos_dim] # MODIFIED: Removed .cuda()
        # Apply dropout to attention output
        # att_out = self.dropout(att_out.cuda())
        att_out = self.dropout(att_out)  # MODIFIED: Removed .cuda() (dropout input is already on device)

        # Project attention output through linear to final logits
        # output = self.linear(att_out.cuda())  # (batch_size, seq_len, |U|)
        output = self.linear(att_out)  # (batch_size, seq_len, |U|) # MODIFIED: Removed .cuda()

        # Get mask of previously activated users to adjust output
        # mask_prev_user = self.get_previous_user_mask(input_seq.cuda(), self.user_num)
        mask_prev_user = self.get_previous_user_mask(input_seq,
                                                     self.user_num)  # input_seq is already on self.device # MODIFIED
        # Add the mask to the output for adjustment
        # output = output.cuda() + mask_prev_user.cuda()
        output = output + mask_prev_user  # MODIFIED: Both are on self.device

        # Reshape to (batch_size * seq_len, |U|) and return
        return output.view(-1, output.size(-1))

    def get_previous_user_mask(self, seq, user_size):
        """ Mask previous activated users."""
        # seq is expected to be on self.device
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask).to(self.device)  # MODIFIED: Move to self.device
        # if seq.is_cuda: # No longer needed, use self.device
        #     previous_mask = previous_mask.cuda()
        masked_seq = previous_mask * seqs.data.float()  # seqs is derived from seq, so it's on self.device

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1, device=self.device)  # MODIFIED: Create on self.device
        # if seq.is_cuda:
        #     PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size,
                              device=self.device)  # MODIFIED: Create on self.device
        # if seq.is_cuda:
        #     ans_tmp = ans_tmp.cuda()
        # masked_seq needs to be long for scatter_
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))  # Ensure masked_seq is long
        # masked_seq = Variable(masked_seq, requires_grad=False) # Variable is deprecated
        masked_seq.requires_grad_(False)  # Use tensor.requires_grad_()
        # print("masked_seq ",masked_seq.size())
        # return masked_seq.cuda()
        return masked_seq  # MODIFIED: Already on self.device

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        # input_*, loss_func, gold are expected to be on self.device from the Runner
        pred = self.forward(input_seq, input_seq_timestamp, history_seq_idx)  # pred will be on self.device

        # gold.contiguous().view(-1) : [bth, max_len-1] -> [bth * (max_len-1)]
        loss = loss_func(pred, gold.contiguous().view(-1))  # Both inputs to loss_func are on self.device

        # Get argmax per row of pred to identify the most probable class per time step; pred.max(1) returns (values, indices), [1] extracts indices.
        pred_choice = pred.max(1)[1]  # pred_choice will be on self.device
        gold_flat = gold.contiguous().view(-1)  # Flatten gold to 1D to match pred.
        n_correct = pred_choice.data.eq(gold_flat.data)  # Compare predictions with gold to get boolean correctness per position.
        # gold.ne(Constants.PAD): boolean mask for non-padding positions.
        mask_correct = gold_flat.ne(Constants.PAD).data  # mask_correct on self.device
        # masked_select(...): select valid (non-padding) entries to avoid counting padding.
        # sum().float(): count correct predictions and cast to float.
        n_correct = n_correct.masked_select(mask_correct).sum().float()  # result is a scalar tensor on self.device
        return loss, n_correct
