import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphBuilder
from utils import Constants
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
        self.hidden_size = args.d_model  # Dimension of word vectors
        self.n_node = data_loader.user_num  # Number of users
        self.pos_dim = args.pos_dim  # Dimension for positional encoding, in the original implement is 8.
        self.dropout = nn.Dropout(args.dropout)  # Dropout layer to prevent overfitting
        self.initial_feature = args.d_model  # Size of initial features

        self.relation_graph = GraphBuilder.build_friendship_network(data_loader)  # load friendship network
        self.hyper_graph_list = GraphBuilder.build_diff_hyper_graph_list(data_loader.cascades,
                                                                         data_loader.timestamps,
                                                                         data_loader.user_num)  # load hypergraph list

        # Initialize components of the model
        self.diff_gnn = HierarchicalGNNWithAttention(self.initial_feature,
                                                     self.hidden_size * 2,
                                                     self.hidden_size,
                                                     dropout=args.dropout)
        self.fri_gnn = GraphNN(self.n_node, self.initial_feature, dropout=args.dropout)  # Friendship
        self.fus = Fusion(self.hidden_size + self.pos_dim)  # Fusion layer
        self.fus2 = Fusion(self.hidden_size)  # Another fusion layer
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)  # Positional embedding
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim,
                                                   n_heads=args.n_heads)  # First attention block
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim,
                                                   n_heads=args.n_heads)  # Second attention block

        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)  # Linear layer for output
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)  # User embeddings
        self.reset_parameters()  # Initialize weights

    def reset_parameters(self):
        # Initialize weights using uniform distribution
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, input_seq, input_seq_timestamp, tgt_idx):
        # 截断输入序列的最后一个元素，保留前面的部分，用于后续的预测操作
        input_seq = input_seq[:, :-1]  # Remove the last object from the tgt
        input_seq_timestamp = input_seq_timestamp[:, :-1]  # Remove last timestamp

        graph = self.relation_graph
        hypergraph_list = self.hyper_graph_list

        # Get hidden representations from friendship network
        hidden = self.dropout(self.fri_gnn(graph))  # fri_gnn([2, edges_num]) -> [n_nodes, hidden_dim]
        # Get memory embeddings from diffusion network
        memory_emb_list = self.diff_gnn(hidden, hypergraph_list)  # {7}

        # Create a mask for padding
        mask = (input_seq == Constants.PAD).cuda()  # [bth, max_len - 1]
        # Prepare positional encoding
        batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).cuda()  # [bth, max_len - 1]
        order_embed = self.dropout(self.pos_embedding(batch_t))  # Positional embeddings [bth, max_len - 1, pos_dim]
        batch_size, max_len = input_seq.size()  # Get batch size and max length

        # Initialize variables for dynamic and cascade embeddings
        zero_vec = torch.zeros_like(input_seq)  # Vector of zeros for masking [bth, max_len-1]
        dyemb = torch.zeros(batch_size, max_len,
                            self.hidden_size).cuda()  # Dynamic embeddings [bth, max_len - 1, hidden_size]
        cas_emb = torch.zeros(batch_size, max_len,
                              self.hidden_size).cuda()  # Cascade embeddings [bth, max_len - 1, hidden_size]

        # Iterate over sorted timestamps in memory embeddings
        for ind, time in enumerate(sorted(memory_emb_list.keys())):
            if ind == 0:
                # For the first timestamp, select inputs based on the current time
                sub_input = torch.where(input_seq_timestamp <= time, input_seq, zero_vec)  # [bth, max_len-1]
                sub_emb = F.embedding(sub_input.cuda(), hidden.cuda())  # [bth, max_len-1, hidden_size]
                temp = sub_input == 0  # Create a mask for zero input, [bth, max_len-1]
                sub_cas = sub_emb.clone()  # Initialize cascade embeddings [bth, max_len-1, hidden_size]
            else:
                # For subsequent timestamps, compare current and previous inputs
                cur = torch.where(input_seq_timestamp <= time, input_seq, zero_vec) - sub_input
                temp = cur == 0  # Mask for current input

                sub_cas = torch.zeros_like(cur)  # Initialize cascade embedding
                sub_cas[~temp] = 1  # Update mask for non-zero current input
                sub_cas = torch.einsum('ij,i->ij', sub_cas, tgt_idx)  # Weight cascade embeddings
                sub_cas = F.embedding(sub_cas.cuda(),
                                      list(memory_emb_list.values())[ind - 1][1].cuda())  # Embed cascade
                sub_emb = F.embedding(cur.cuda(),
                                      list(memory_emb_list.values())[ind - 1][0].cuda())  # Embed current input
                sub_input = cur + sub_input  # Update sub_input for next iteration

            # Update embeddings by zeroing out masked values
            sub_cas[temp] = 0
            sub_emb[temp] = 0
            dyemb += sub_emb  # Accumulate dynamic embeddings, [bth, max_len-1, hidden_size]
            cas_emb += sub_cas  # Accumulate cascade embeddings, [bth, max_len-1, hidden_size]

            if ind == len(memory_emb_list) - 1:
                # For the last timestamp, finalize the embeddings
                sub_input = input_seq - sub_input  # Calculate remaining input
                temp = sub_input == 0  # Mask for remaining input

                sub_cas = torch.zeros_like(sub_input)  # Initialize for last input
                sub_cas[~temp] = 1  # Update mask
                sub_cas = torch.einsum('ij,i->ij', sub_cas, tgt_idx)  # Weight final cascade embeddings
                sub_cas = F.embedding(sub_cas.cuda(),
                                      list(memory_emb_list.values())[ind - 1][1].cuda())  # Embed final cascade
                sub_cas[temp] = 0  # Zero out masked values
                sub_emb = F.embedding(sub_input.cuda(),
                                      list(memory_emb_list.values())[ind][0].cuda())  # Embed remaining input
                sub_emb[temp] = 0  # Zero out masked values

                dyemb += sub_emb  # Accumulate final dynamic embeddings
                cas_emb += sub_cas  # Accumulate final cascade embeddings

        # Concatenate dynamic and order embeddings
        diff_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()  # [bth, max_len-1, pos_dim+hidden_size]
        # Concatenate user embeddings with positional encoding
        fri_embed = torch.cat(
            [F.embedding(input_seq.cuda(),
                         hidden.cuda()),
             order_embed], dim=-1).cuda()  # [bth, max_len-1, pos_dim+hidden_size]

        # Apply attention mechanisms
        diff_att_out = self.decoder_attention1(diff_embed,
                                               diff_embed,
                                               diff_embed,
                                               mask=mask)  # [bth, max_len-1, pos_dim+hidden_size]
        diff_att_out = self.dropout(diff_att_out.cuda())  # Dropout on attention output

        fri_att_out = self.decoder_attention2(fri_embed,
                                              fri_embed,
                                              fri_embed,
                                              mask=mask)  # [bth, max_len-1, pos_dim+hidden_size]
        fri_att_out = self.dropout(fri_att_out.cuda())  # Dropout on attention output

        # Fuse the attention outputs
        att_out = self.fus(diff_att_out, fri_att_out)  # [bth, max_len-1, pos_dim+hidden_size]

        # Combine user and cascade outputs
        output_u = self.linear2(att_out.cuda())  # [bth, max_len-1, node_num]
        mask = self.get_previous_user_mask(input_seq.cpu(), self.n_node)  # Get user mask

        return (output_u + mask).view(-1, output_u.size(-1))  # [bth * max_len-1, node_num]

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

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        pred = self.forward(input_seq, input_seq_timestamp, history_seq_idx)

        # gold.contiguous().view(-1) : [bth, max_len-1] -> [bth * (max_len-1)]
        loss = loss_func(pred, gold.contiguous().view(-1))
        # 获取 pred 中每行的最大值的索引，表示模型认为每个时间步最可能的类别,pred.max(1) 返回一个包含最大值和索引的元组，而 [1] 表示取出索引部分。
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)  # 将 gold 转换为一维数组，确保它与 pred 的展平形状一致。
        n_correct = pred.data.eq(gold.data)  # 比较 pred 和 gold，返回一个布尔数组，表示每个位置是否预测正确。
        # gold.ne(Constants.PAD): 生成一个布尔数组，标记 gold 中不是填充符的部分。
        # masked_select(...): 只选择有效的（非填充）预测，确保不会将填充位置计入正确预测。
        # sum().float(): 最后计算正确预测的数量，并将其转换为浮点数。
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
        return loss, n_correct

