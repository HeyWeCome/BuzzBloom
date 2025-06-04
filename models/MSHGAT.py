import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Assuming layers and utils are in the correct path
from layers import GraphBuilder
from utils import Constants  # Make sure Constants.PAD is defined
from torch.autograd import Variable
from layers.Commons import HierarchicalGNNWithAttention, GraphNN, Fusion
from layers.TransformerBlock import TransformerBlock


class MSHGAT(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--pos_dim', type=int, default=8,
                            help='Dimension for positional encoding, in the original implement is 8.')
        parser.add_argument('--n_heads', type=int, default=8,
                            help='Number of the attention head, in the original implement is 8.')
        return parser

    def __init__(self, args, data_loader):
        super(MSHGAT, self).__init__()
        self.device = args.device
        self.hidden_size = args.d_model
        self.n_node = data_loader.user_num
        self.pos_dim = args.pos_dim
        self.dropout = nn.Dropout(args.dropout)
        self.initial_feature = args.d_model

        relation_graph_cpu = GraphBuilder.build_friendship_network(data_loader)
        if isinstance(relation_graph_cpu, torch.Tensor):
            self.relation_graph = relation_graph_cpu.to(self.device)
        else:
            self.relation_graph = relation_graph_cpu  # Assuming GraphNN handles device or it's not a tensor needing .to()

        self.hyper_graph_list = GraphBuilder.build_diff_hyper_graph_list(data_loader.cascades,
                                                                         data_loader.timestamps,
                                                                         data_loader.user_num)

        self.diff_gnn = HierarchicalGNNWithAttention(self.initial_feature,
                                                     self.hidden_size * 2,
                                                     self.hidden_size,
                                                     dropout=args.dropout)
        self.fri_gnn = GraphNN(self.n_node, self.initial_feature, dropout=args.dropout)
        self.fus = Fusion(self.hidden_size + self.pos_dim)
        self.fus2 = Fusion(self.hidden_size)  # This fus2 is defined but not used in the forward pass shown
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim,
                                                   n_heads=args.n_heads)
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim,
                                                   n_heads=args.n_heads)
        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)  # Main user embedding
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, input_seq, input_seq_timestamp, tgt_idx):
        input_seq_orig = input_seq.to(self.device)  # Keep original for full length if needed later
        input_seq_timestamp_orig = input_seq_timestamp.to(self.device)
        if isinstance(tgt_idx, torch.Tensor):
            tgt_idx = tgt_idx.to(self.device)
        # Else: tgt_idx might be a list or other type, handle as per its usage

        input_seq = input_seq_orig[:, :-1]
        input_seq_timestamp = input_seq_timestamp_orig[:, :-1]

        graph = self.relation_graph
        hypergraph_list = self.hyper_graph_list

        hidden = self.dropout(self.fri_gnn(graph))  # Global user embeddings from friendship graph
        memory_emb_list = self.diff_gnn(hidden, hypergraph_list)  # Time-aware embeddings from diffusion hypergraphs

        mask = (input_seq == Constants.PAD)  # Mask for attention, already on device due to input_seq
        batch_t = torch.arange(input_seq.size(1), device=self.device).expand(input_seq.size())
        order_embed = self.dropout(self.pos_embedding(batch_t))
        batch_size, max_len = input_seq.size()

        zero_vec = torch.zeros_like(input_seq)  # For masking, on device
        dyemb = torch.zeros(batch_size, max_len, self.hidden_size, device=self.device)
        cas_emb = torch.zeros(batch_size, max_len, self.hidden_size, device=self.device)

        # This variable tracks the cumulative input processed based on timestamps
        cumulative_processed_input_ids = torch.zeros_like(input_seq, device=self.device)

        sorted_timestamps = sorted(memory_emb_list.keys())

        for ind, time_key in enumerate(sorted_timestamps):
            # Embeddings from GNN at specific snapshot 'time_key'
            # These are WEIGHTS for F.embedding, not features of current input_seq yet
            snapshot_global_emb_weights = memory_emb_list[time_key][0]
            snapshot_cascade_emb_weights = memory_emb_list[time_key][1]

            # Variables for current segment's features
            current_segment_dynamic_features = None
            current_segment_cascade_features = None
            current_segment_mask = None  # Mask for current segment (where current_segment_input_ids == 0)

            if ind == 0:
                # Users active up to the first timestamp 'time_key'
                current_segment_input_ids = torch.where(input_seq_timestamp <= time_key, input_seq, zero_vec)
                current_segment_mask = (current_segment_input_ids == 0)

                # For t=0, dynamic features come from the global 'hidden' state (friendship GNN)
                current_segment_dynamic_features = F.embedding(current_segment_input_ids, hidden)
                # For t=0, cascade features are cloned from dynamic features
                current_segment_cascade_features = current_segment_dynamic_features.clone()

                cumulative_processed_input_ids = current_segment_input_ids.clone()
            else:  # ind > 0
                # Embeddings from GNN snapshot at *previous* timestamp
                prev_time_key = sorted_timestamps[ind - 1]
                prev_snapshot_global_emb_weights = memory_emb_list[prev_time_key][0]
                prev_snapshot_cascade_emb_weights = memory_emb_list[prev_time_key][1]

                # User IDs newly active in this segment (between prev_time_key and time_key)
                current_segment_input_ids = torch.where(input_seq_timestamp <= time_key, input_seq,
                                                        zero_vec) - cumulative_processed_input_ids
                current_segment_mask = (current_segment_input_ids == 0)

                # Dynamic features for this segment from prev. global GNN snapshot
                current_segment_dynamic_features = F.embedding(current_segment_input_ids,
                                                               prev_snapshot_global_emb_weights)

                # Cascade features for this segment from prev. cascade GNN snapshot
                # First, prepare indices for F.embedding:
                cascade_indices_for_this_segment = torch.zeros_like(current_segment_input_ids, device=self.device,
                                                                    dtype=torch.long)

                # Active users in this segment (non-zero in current_segment_input_ids) get index 1 or from tgt_idx
                # Inactive users (zero in current_segment_input_ids, i.e. current_segment_mask is True) get index 0
                if isinstance(tgt_idx, torch.Tensor) and tgt_idx.numel() > 0:
                    # Assuming tgt_idx are [batch_size] specific base indices for each cascade.
                    # We need to expand tgt_idx to [batch_size, max_len] to select based on activity.
                    expanded_tgt_idx = tgt_idx.view(-1, 1).expand_as(cascade_indices_for_this_segment).long()
                    cascade_indices_for_this_segment = torch.where(~current_segment_mask, expanded_tgt_idx,
                                                                   torch.tensor(0, device=self.device,
                                                                                dtype=torch.long))
                else:
                    # Default: active users in segment get index 1, inactive get 0.
                    cascade_indices_for_this_segment[~current_segment_mask] = 1

                # THE LINE THAT CAUSED ERROR (equivalent): sub_cas = F.embedding(sub_cas_for_embedding, prev_memory_cascade_emb)
                current_segment_cascade_features = F.embedding(cascade_indices_for_this_segment,
                                                               prev_snapshot_cascade_emb_weights)

                cumulative_processed_input_ids += current_segment_input_ids

            # Mask out parts of segment features that were padding, then accumulate
            if current_segment_dynamic_features is not None and current_segment_mask is not None:
                current_segment_dynamic_features[current_segment_mask] = 0
                dyemb += current_segment_dynamic_features

            if current_segment_cascade_features is not None and current_segment_mask is not None:
                current_segment_cascade_features[current_segment_mask] = 0
                cas_emb += current_segment_cascade_features

            # Processing for the "final" segment of users (those active *after* the last timestamp in memory_emb_list)
            if ind == len(sorted_timestamps) - 1:
                # Users not yet covered by any timestamped GNN snapshots
                remaining_input_ids = input_seq - cumulative_processed_input_ids
                remaining_mask = (remaining_input_ids == 0)

                # Dynamic features from the GNN snapshot at 'time_key' (current, which is the last available)
                final_segment_dynamic_features = F.embedding(remaining_input_ids, snapshot_global_emb_weights)
                final_segment_dynamic_features[remaining_mask] = 0
                dyemb += final_segment_dynamic_features

                # Cascade features: original code used (ind-1) for weights.
                # These weights should correspond to GNN state *before* the 'remaining_input_ids' became active.
                cascade_weights_for_final_segment = None
                if len(sorted_timestamps) == 1:  # Only one timestamp in memory, ind is 0
                    cascade_weights_for_final_segment = snapshot_cascade_emb_weights  # use current
                else:  # More than one timestamp, ind is last. ind-1 is the one before 'time_key'
                    # This logic matches original: list(memory_emb_list.values())[ind - 1][1]
                    cascade_weights_for_final_segment = memory_emb_list[sorted_timestamps[ind - 1]][1]

                final_cascade_indices = torch.zeros_like(remaining_input_ids, device=self.device, dtype=torch.long)
                if isinstance(tgt_idx, torch.Tensor) and tgt_idx.numel() > 0:
                    expanded_tgt_idx = tgt_idx.view(-1, 1).expand_as(final_cascade_indices).long()
                    final_cascade_indices = torch.where(~remaining_mask, expanded_tgt_idx,
                                                        torch.tensor(0, device=self.device, dtype=torch.long))
                else:
                    final_cascade_indices[~remaining_mask] = 1

                final_segment_cascade_features = F.embedding(final_cascade_indices, cascade_weights_for_final_segment)
                final_segment_cascade_features[remaining_mask] = 0
                cas_emb += final_segment_cascade_features

        diff_embed = torch.cat([dyemb, order_embed], dim=-1)
        fri_embed = torch.cat([F.embedding(input_seq, hidden), order_embed], dim=-1)

        diff_att_out = self.decoder_attention1(diff_embed, diff_embed, diff_embed, mask=mask)
        diff_att_out = self.dropout(diff_att_out)

        fri_att_out = self.decoder_attention2(fri_embed, fri_embed, fri_embed, mask=mask)
        fri_att_out = self.dropout(fri_att_out)

        att_out = self.fus(diff_att_out, fri_att_out)
        output_u = self.linear2(att_out)

        # Mask for previous users in the sequence
        # Ensure input_seq for this mask is the one used in the main logic (i.e., input_seq[:, :-1])
        prev_user_mask = self.get_previous_user_mask(input_seq, self.n_node, self.device)

        return (output_u + prev_user_mask).view(-1, output_u.size(-1))

    def get_previous_user_mask(self, seq, user_size, device):
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        # seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1)) # This was for tril logic
        # Simpler: iterate and build mask
        # However, the tril logic is standard for masking future items, let's stick to making it work on device.

        # Create tril on the target device
        # previous_mask_np = np.tril(np.ones(prev_shape, dtype=np.float32), k=-1) # k=-1 to mask current token itself from seeing itself if needed
        # For masking previous tokens (users already appeared):
        batch_size, seq_len = seq.shape
        mask_tensor = torch.ones((batch_size, seq_len, user_size), device=device) * float(
            '-inf')  # Start with all masked

        for b in range(batch_size):
            for t in range(seq_len):
                # For position t, unmask users that have appeared in seq[b, :t]
                # No, this should mask users that *have* appeared.
                # The original scatter approach is more direct for "masking already activated users".
                # Scatter -1000 to indices that have appeared.

                # Re-implementing the scatter approach carefully:
                # For each item in the sequence, gather all previous items.
                # Example: seq = [[1,2,3],[4,5,1]]
                # t=0: mask nothing (or PAD)
                # t=1: mask seq[b,0]
                # t=2: mask seq[b,0], seq[b,1]

                # The original `get_previous_user_mask` creates a mask of shape (batch, seq_len, user_size)
                # where `mask[b, t, user_idx] = -1000` if `user_idx` has appeared in `seq[b, 0...t-1]`.
                # (Or 0...t including current, original code used `np.tril(np.ones(prev_shape))`, which includes diagonal)

                # Let's use the original scatter approach, ensuring device compatibility
                if t > 0:  # Only apply mask if there are previous tokens
                    users_to_mask_at_t = seq[b, :t]
                    # Remove PADs from users_to_mask, as PAD index 0 should not be explicitly scattered to unless intended
                    valid_users_to_mask = users_to_mask_at_t[users_to_mask_at_t != Constants.PAD]
                    if valid_users_to_mask.numel() > 0:
                        mask_tensor[b, t, valid_users_to_mask.long()] = float('-1000')  # Mask these users
                # Always mask PAD token index if it's a user
                mask_tensor[b, t, Constants.PAD] = float('-1000')

        # The original code used tril and scatter, which is efficient. Let's revert to that logic, ensuring device placement.
        seqs_expanded = seq.unsqueeze(2).expand(batch_size, seq_len,
                                                seq_len)  # [B, L, L] where seqs_expanded[b,t,:] = seq[b,t]
        # No, this is not right. Original: seqs = seq.repeat(1, 1, seq.size(1)).view(...)
        # This made seqs[b,t,k] = seq[b,k] -- i.e., for each target position t, you have the whole sequence.

        # Corrected replication of original scatter logic for get_previous_user_mask
        tril_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                               diagonal=0)  # Include current token
        # tril_mask is [L,L]. We want for each [B, T_target, T_source]

        ans_tmp = torch.zeros(batch_size, seq_len, user_size, device=device)  # Default is 0 (no Veränderung)
        # For each position `t` in the sequence, we want to mask users that appeared in `seq[:, 0...t]`
        for t in range(seq_len):
            users_seen_up_to_t = seq[:, :t + 1]  # [B, t+1]
            # Create a mask for current position t based on users_seen_up_to_t
            # This needs to be efficient. Scatter is good.
            # For each batch element b, at time t, scatter -1000 for users in users_seen_up_to_t[b]
            for b_idx in range(batch_size):
                # Users seen by batch b up to time t (unique and non-PAD)
                unique_users = torch.unique(users_seen_up_to_t[b_idx][users_seen_up_to_t[b_idx] != Constants.PAD])
                if unique_users.numel() > 0:
                    ans_tmp[b_idx, t, unique_users.long()] = float('-1000')
            # Always mask PAD user
            ans_tmp[:, t, Constants.PAD] = float('-1000')

        ans_tmp.requires_grad_(False)
        return ans_tmp

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        gold = gold.to(self.device)
        pred = self.forward(input_seq, input_seq_timestamp, history_seq_idx)

        loss = loss_func(pred, gold.contiguous().view(-1))

        pred_labels = pred.max(1)[1]
        gold_flat = gold.contiguous().view(-1)

        non_pad_mask = gold_flat.ne(Constants.PAD)  # .data is not needed here
        n_correct = pred_labels.eq(gold_flat).masked_select(non_pad_mask).sum().float()

        return loss, n_correct
