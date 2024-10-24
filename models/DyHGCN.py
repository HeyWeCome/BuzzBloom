import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from layers import GraphBuilder
from utils import Constants
from torch.autograd import Variable
from layers.Commons import DynamicGraphNN, GraphNN, Fusion, TimeAttention
from layers.TransformerBlock import TransformerBlock


class DyHGCN(nn.Module):
    """
    Only implement DyHGCN-S, because its performance is better.
    """
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
        self.user_num = data_loader.user_num
        self.embedding_size = args.d_model
        self.pos_dim = args.pos_dim
        self.n_heads = args.n_heads
        self.time_step_split = args.time_step_split
        self.dropout = nn.Dropout(args.dropout)
        self.drop_timestamp = nn.Dropout(args.dropout)

        # In the original paper, the dropout is 0.5
        self.gnn_layer = GraphNN(self.user_num, self.embedding_size, dropout=0.5)
        self.gnn_diffusion_layer = DynamicGraphNN(self.user_num, self.embedding_size, self.time_step_split)
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)

        self.time_attention = TimeAttention(self.time_step_split, self.embedding_size)
        self.decoder_attention = TransformerBlock(input_size=self.embedding_size + self.pos_dim, n_heads=self.n_heads)
        self.linear = nn.Linear(self.embedding_size + self.pos_dim, self.user_num)

        self.relation_graph = GraphBuilder.build_friendship_network(data_loader)  # load friendship network
        self.diffusion_graph = GraphBuilder.build_dynamic_heterogeneous_graph(data_loader, self.time_step_split)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def forward(self, input, input_timestamp, tgt_idx):
        input = input[:, :-1]
        mask = (input == Constants.PAD)

        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))

        batch_size, max_len = input.size()
        dyemb = torch.zeros(batch_size, max_len, self.user_num).cuda()
        input_timestamp = input_timestamp[:, :-1]
        step_len = 5

        dynamic_node_emb_dict = self.gnn_diffusion_layer(self.diffusion_graph)  # input, input_timestamp, diffusion_graph)
        dyemb_timestamp = torch.zeros(batch_size, max_len).long()

        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val] = i
        latest_timestamp = dynamic_node_emb_dict_time[-1]
        for t in range(0, max_len, step_len):
            try:
                la_timestamp = torch.max(input_timestamp[:, t:t + step_len]).item()
                if la_timestamp < 1:
                    break
                latest_timestamp = la_timestamp
            except Exception:
                pass

            res_index = len(dynamic_node_emb_dict_time_dict) - 1
            for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                if val <= latest_timestamp:
                    res_index = i
                    continue
                else:
                    break
            dyemb_timestamp[:, t:t + step_len] = res_index

        dyuser_emb_list = list()
        for val in sorted(dynamic_node_emb_dict.keys()):
            dyuser_emb_sub = F.embedding(input.cuda(), dynamic_node_emb_dict[val].cuda()).unsqueeze(2)
            dyuser_emb_list.append(dyuser_emb_sub)
        dyuser_emb = torch.cat(dyuser_emb_list, dim=2)

        dyemb = self.time_attention(dyemb_timestamp.cuda(), dyuser_emb.cuda())
        dyemb = self.dropout(dyemb)

        final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()  # dynamic_node_emb
        att_out = self.decoder_attention(final_embed.cuda(), final_embed.cuda(), final_embed.cuda(), mask=mask.cuda())
        att_out = self.dropout(att_out.cuda())

        output = self.linear(att_out.cuda())  # (bsz, user_len, |U|)
        mask = self.get_previous_user_mask(input.cuda(), self.user_num)
        output = output.cuda() + mask.cuda()

        return output.view(-1, output.size(-1))

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
