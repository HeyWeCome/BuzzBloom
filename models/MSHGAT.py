import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import Constants
from torch.autograd import Variable
from layers.Commons import HierarchicalGNNWithAttention, GraphNN, Fusion
from layers.TransformerBlock import TransformerBlock


class MSHGAT(nn.Module):

    # @staticmethod
    # def parse_model_args(parser):
    #     parser.add_argument('--embedding_size', type=int, default=64,
    #                         help='Size of embedding vectors.')
    #     return parser

    def __init__(self, args, data_loader):
        super(MSHGAT, self).__init__()
        self.hidden_size = args.d_model  # Dimension of word vectors
        self.n_node = data_loader.user_num  # Number of users
        self.pos_dim = 8  # Dimension for positional encoding
        self.dropout = nn.Dropout(args.dropout)  # Dropout layer to prevent overfitting
        self.initial_feature = args.d_model  # Size of initial features

        # Initialize components of the model
        self.diff_gnn = HierarchicalGNNWithAttention(self.initial_feature,
                                                     self.hidden_size * 2,
                                                     self.hidden_size,
                                                     dropout=args.dropout)
        self.fri_gnn = GraphNN(self.n_node, self.initial_feature, dropout=args.dropout)  # Friendship
        self.fus = Fusion(self.hidden_size + self.pos_dim)  # Fusion layer
        self.fus2 = Fusion(self.hidden_size)  # Another fusion layer
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)  # Positional embedding
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim,
                                                   n_heads=8)  # First attention block
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim,
                                                   n_heads=8)  # Second attention block

        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)  # Linear layer for output
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)  # User embeddings
        self.reset_parameters()  # Initialize weights


    def reset_parameters(self):
        # Initialize weights using uniform distribution
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list):
        tgt = tgt[:, :-1]  # Remove the last object from the tgt
        tgt_timestamp = tgt_timestamp[:, :-1]  # Remove last timestamp

        # Get hidden representations from friendship network
        hidden = self.dropout(self.fri_gnn(graph))
        # Get memory embeddings from diffusion network
        memory_emb_list = self.diff_gnn(hidden, hypergraph_list)

        # Create a mask for padding
        mask = (tgt == Constants.PAD)
        # Prepare positional encoding
        batch_t = torch.arange(tgt.size(1)).expand(tgt.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))  # Positional embeddings
        batch_size, max_len = tgt.size()  # Get batch size and max length

        # Initialize variables for dynamic and cascade embeddings
        zero_vec = torch.zeros_like(tgt)  # Vector of zeros for masking
        dyemb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()  # Dynamic embeddings
        cas_emb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()  # Cascade embeddings

        # Iterate over sorted timestamps in memory embeddings
        for ind, time in enumerate(sorted(memory_emb_list.keys())):
            if ind == 0:
                # For the first timestamp, select inputs based on the current time
                sub_input = torch.where(tgt_timestamp <= time, tgt, zero_vec)
                sub_emb = F.embedding(sub_input.cuda(), hidden.cuda())  # Embed the selected input
                temp = sub_input == 0  # Create a mask for zero input
                sub_cas = sub_emb.clone()  # Initialize cascade embeddings
            else:
                # For subsequent timestamps, compare current and previous inputs
                cur = torch.where(tgt_timestamp <= time, tgt, zero_vec) - sub_input
                temp = cur == 0  # Mask for current input

                sub_cas = torch.zeros_like(cur)  # Initialize cascade embedding
                sub_cas[~temp] = 1  # Update mask for non-zero current input
                sub_cas = torch.einsum('ij,i->ij', sub_cas, tgt_idx)  # Weight cascade embeddings
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())  # Embed cascade
                sub_emb = F.embedding(cur.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())  # Embed current input
                sub_input = cur + sub_input  # Update sub_input for next iteration

            # Update embeddings by zeroing out masked values
            sub_cas[temp] = 0
            sub_emb[temp] = 0
            dyemb += sub_emb  # Accumulate dynamic embeddings
            cas_emb += sub_cas  # Accumulate cascade embeddings

            if ind == len(memory_emb_list) - 1:
                # For the last timestamp, finalize the embeddings
                sub_input = tgt - sub_input  # Calculate remaining input
                temp = sub_input == 0  # Mask for remaining input

                sub_cas = torch.zeros_like(sub_input)  # Initialize for last input
                sub_cas[~temp] = 1  # Update mask
                sub_cas = torch.einsum('ij,i->ij', sub_cas, tgt_idx)  # Weight final cascade embeddings
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())  # Embed final cascade
                sub_cas[temp] = 0  # Zero out masked values
                sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind][0].cuda())  # Embed remaining input
                sub_emb[temp] = 0  # Zero out masked values

                dyemb += sub_emb  # Accumulate final dynamic embeddings
                cas_emb += sub_cas  # Accumulate final cascade embeddings

        # Concatenate dynamic and order embeddings
        diff_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()
        # Concatenate user embeddings with positional encoding
        fri_embed = torch.cat([F.embedding(tgt.cuda(), hidden.cuda()), order_embed], dim=-1).cuda()

        # Apply attention mechanisms
        diff_att_out = self.decoder_attention1(diff_embed.cuda(), diff_embed.cuda(), diff_embed.cuda(), mask=mask.cuda())
        diff_att_out = self.dropout(diff_att_out.cuda())  # Dropout on attention output

        fri_att_out = self.decoder_attention2(fri_embed.cuda(), fri_embed.cuda(), fri_embed.cuda(), mask=mask.cuda())
        fri_att_out = self.dropout(fri_att_out.cuda())  # Dropout on attention output

        # Fuse the attention outputs
        att_out = self.fus(diff_att_out, fri_att_out)

        # Combine user and cascade outputs
        output_u = self.linear2(att_out.cuda())  # Final output for users
        mask = self.get_previous_user_mask(tgt.cpu(), self.n_node)  # Get user mask

        return (output_u + mask).view(-1, output_u.size(-1)).cuda()  # Return the output reshaped


    def get_previous_user_mask(self, seq, user_size):
        """ Mask previous activated users."""
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()
        masked_seq = previous_mask * seqs.data.float()

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
        masked_seq = Variable(masked_seq, requires_grad=False)
        # print("masked_seq ",masked_seq.size())
        return masked_seq.cuda()
