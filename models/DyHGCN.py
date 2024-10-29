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

        # self.relation_graph = GraphBuilder.build_friendship_network(data_loader)  # load friendship network
        self.diffusion_graph = GraphBuilder.build_dynamic_heterogeneous_graph(data_loader, self.time_step_split)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def forward(self, input_seq, input_timestamp, tgt_idx):
        # 截断输入序列的最后一个元素，保留前面的部分，用于后续的预测操作
        input_seq = input_seq[:, :-1]  # [bth, seq_len]
        input_timestamp = input_timestamp[:, :-1]  # [bth, seq_len]

        # 创建掩码，用于标记填充位置（Constants.PAD）为True，其他位置为False
        mask = (input_seq == Constants.PAD)  # [bth, seq_len]

        # 生成批次中的时间步索引，并通过位置嵌入层获取位置嵌入
        batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).cuda()  # [bth, seq_len]
        # 对位置嵌入应用dropout，以防止过拟合
        order_embed = self.dropout(self.pos_embedding(batch_t))  # [bth, seq_len, pos_dim]

        # 获取批次大小和输入序列的最大长度
        batch_size, max_len = input_seq.size()

        # 初始化dyemb张量，用于存储动态节点嵌入，尺寸为 (batch_size, max_len, self.user_num)
        dyemb = torch.zeros(batch_size, max_len, self.user_num).cuda()

        # 定义时间步长，用于动态嵌入的更新
        step_len = 5

        # 计算每个时间步的动态节点嵌入图
        dynamic_node_emb_dict = self.gnn_diffusion_layer(self.diffusion_graph)  # 8个时间段

        # 初始化dyemb_timestamp张量，用于存储每个时间步对应的动态嵌入索引
        dyemb_timestamp = torch.zeros(batch_size, max_len).long()  # [bth, seq_len]

        # 获取动态嵌入字典中所有的时间戳，并创建一个映射字典
        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val] = i

        # 获取字典中最新的时间戳
        latest_timestamp = dynamic_node_emb_dict_time[-1]

        # 遍历输入序列的时间步，并为每个时间片段选择合适的动态嵌入
        for t in range(0, max_len, step_len):
            try:
                # 获取当前时间片段的最大时间戳
                la_timestamp = torch.max(input_timestamp[:, t:t + step_len]).item()
                # 如果最大时间戳小于1，说明没有有效的时间戳数据，跳出循环
                if la_timestamp < 1:
                    break
            except Exception:
                pass

            # 根据时间戳确定在动态嵌入字典中的索引位置
            res_index = len(dynamic_node_emb_dict_time_dict) - 1
            for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                # 找到小于或等于最新时间戳的最大索引值
                if val <= latest_timestamp:
                    res_index = i
                    continue
                else:
                    break
            # 将当前时间步的动态嵌入索引更新到dyemb_timestamp中
            dyemb_timestamp[:, t:t + step_len] = res_index

        # 创建一个列表，用于存储每个时间步的用户嵌入
        dyuser_emb_list = list()
        for val in sorted(dynamic_node_emb_dict.keys()):
            # 使用embedding函数从输入序列中获取对应时间步的用户嵌入
            dyuser_emb_sub = F.embedding(input_seq.cuda(), dynamic_node_emb_dict[val].cuda()).unsqueeze(2)
            dyuser_emb_list.append(dyuser_emb_sub)

        # 将所有时间步的用户嵌入沿第二维拼接，形成完整的用户嵌入张量
        dyuser_emb = torch.cat(dyuser_emb_list, dim=2)  # [bth, seq_len, time_step, hidden_size]

        # 通过时间注意力机制融合不同时间步的用户嵌入
        dyemb = self.time_attention(dyemb_timestamp.cuda(), dyuser_emb.cuda())  # [bth, seq_len, hidden_size]
        # 对动态嵌入应用dropout
        dyemb = self.dropout(dyemb)

        # 将动态嵌入和顺序嵌入沿最后一维拼接，形成最终的嵌入表示
        final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()  # [bth, seq_len, hidden_size+pos_dim]

        # 使用decoder_attention模块计算自注意力输出，使用掩码处理填充位置
        att_out = self.decoder_attention(final_embed.cuda(),
                                         final_embed.cuda(),
                                         final_embed.cuda(),
                                         mask=mask.cuda())  # [batch_size, seq_len, hidden_size+pos_dim]
        # 对注意力输出应用dropout
        att_out = self.dropout(att_out.cuda())

        # 通过线性层将注意力输出转换为最终的输出表示
        output = self.linear(att_out.cuda())  # (batch_size, seq_len, |U|)

        # 获取之前用户的掩码，用于调整输出结果
        mask = self.get_previous_user_mask(input_seq.cuda(), self.user_num)
        # 将掩码添加到输出中，进行适当的调整
        output = output.cuda() + mask.cuda()

        # 将输出调整为 (batch_size * user_len, |U|) 的形状并返回
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
