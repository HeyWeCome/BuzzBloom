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
    Only implement DyHGCN-S, because its performance is better.
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
        # 截断输入序列的最后一个元素，保留前面的部分，用于后续的预测操作
        input_seq = input_seq[:, :-1]  # [bth, seq_len]
        input_timestamp = input_timestamp[:, :-1]  # [bth, seq_len]

        # 创建掩码，用于标记填充位置（Constants.PAD）为True，其他位置为False
        mask = (input_seq == Constants.PAD)  # [bth, seq_len]

        # 生成批次中的时间步索引，并通过位置嵌入层获取位置嵌入
        # batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).cuda()  # [bth, seq_len]
        batch_t = torch.arange(input_seq.size(1), device=self.device).expand(
            input_seq.size())  # MODIFIED: Create on self.device
        # 对位置嵌入应用dropout，以防止过拟合
        order_embed = self.dropout(self.pos_embedding(batch_t))  # [bth, seq_len, pos_dim]

        # 获取批次大小和输入序列的最大长度
        batch_size, max_len = input_seq.size()

        # 初始化dyemb张量，用于存储动态节点嵌入，尺寸为 (batch_size, max_len, self.user_num)
        # dyemb = torch.zeros(batch_size, max_len, self.user_num).cuda()
        # dyemb is reassigned later by self.time_attention, so this initialization might not be strictly needed
        # If it were used directly before reassignment, it should be on device.
        # For now, commenting out as it seems unused before being overwritten.
        # dyemb_init_placeholder = torch.zeros(batch_size, max_len, self.user_num, device=self.device) # MODIFIED: Create on self.device

        # 定义时间步长，用于动态嵌入的更新
        step_len = 5

        # 计算每个时间步的动态节点嵌入图
        # dynamic_node_emb_dict's values (embeddings) are expected to be on self.device
        # as gnn_diffusion_layer is an nn.Module and its parameters should be on self.device.
        dynamic_node_emb_dict = self.gnn_diffusion_layer(self.diffusion_graph)  # 8个时间段

        # 初始化dyemb_timestamp张量，用于存储每个时间步对应的动态嵌入索引
        dyemb_timestamp = torch.zeros(batch_size, max_len, dtype=torch.long,
                                      device=self.device)  # [bth, seq_len] # MODIFIED: Create on self.device, ensure long dtype

        # 获取动态嵌入字典中所有的时间戳，并创建一个映射字典
        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val] = i

        # 获取字典中最新的时间戳
        latest_timestamp_val = dynamic_node_emb_dict_time[
            -1] if dynamic_node_emb_dict_time else 0  # ADDED: Handle empty list

        # 遍历输入序列的时间步，并为每个时间片段选择合适的动态嵌入
        """举例
        dynamic_node_emb_dict_time_dict = {5: emb_5, 10: emb_10, 15: emb_15}
            字典中最新的时间戳 latest_timestamp_val=15
        input_timestamp = torch.tensor([
            [3, 6, 8, 9, 14, 16],   # 序列 1 的时间步
            [1, 4, 7, 12, 13, 17]   # 序列 2 的时间步
        ])
        每次处理2个时间步：
            第一次：torch.tensor([[3, 6], [1, 4]])
                当前时间片段的最大时间戳 la_timestamp = 6
                在字典中查找<=6的最大时间戳，即val=5，i=0，res_index=i=0
            第二次：torch.tensor([[8, 9], [7, 12]])
                la_timestamp = 12
                在字典中查找<=12的最大时间戳，即val=10，i=1，res_index=i=1
            第三次：torch.tensor([[14, 16], [13, 17]])
                la_timestamp = 17
                在字典中查找<=17的最大时间戳，即val=15，i=2，res_index=i=2
        得到dyemb_timestamp = torch.tensor([
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2]
            ])
        """
        current_latest_processed_ts = latest_timestamp_val  # Use the initial latest from dict
        for t in range(0, max_len, step_len):
            try:
                # 获取当前时间片段的最大时间戳
                current_chunk_timestamp = input_timestamp[:, t:t + step_len]
                # Ensure there are non-PAD values before taking max if PAD is not a valid timestamp
                valid_timestamps_in_chunk = current_chunk_timestamp[
                    current_chunk_timestamp != Constants.PAD_TIME if hasattr(Constants,
                                                                             'PAD_TIME') else current_chunk_timestamp >= 0]  # Assuming PAD_TIME or negative for PADs
                if valid_timestamps_in_chunk.numel() > 0:
                    la_timestamp = torch.max(valid_timestamps_in_chunk).item()
                    # 如果最大时间戳小于1 (or some other threshold if 0 is valid timestamp), 说明没有有效的时间戳数据
                    if la_timestamp < 1 and t == 0:  # Check for meaningful timestamps, especially for the first chunk
                        # This break condition might be too strict if 0 is a valid start time.
                        # Consider if this logic needs adjustment based on data specifics.
                        # break
                        pass  # Continue to assign based on available dict keys
                    current_latest_processed_ts = la_timestamp  # 将当前时间片段的最大时间戳赋给最新时间戳
                # else:
                # If chunk is all PADs or invalid, what should be the index?
                # Default to the last known good index or first/last dict index.
                # Current logic will use previous current_latest_processed_ts
                # pass
            except Exception as e:
                # logging.warning(f"Error processing timestamp chunk at t={t}: {e}")
                pass  # Keep previous current_latest_processed_ts

            # 根据时间戳确定在动态嵌入字典中的索引位置
            res_index = len(dynamic_node_emb_dict_time_dict) - 1 if dynamic_node_emb_dict_time_dict else 0
            # 找到小于或等于最新时间戳的最大索引值
            # Ensure dynamic_node_emb_dict_time_dict is not empty
            if dynamic_node_emb_dict_time_dict:
                for i, val_key in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                    # 不断遍历，直至val_key不再小于等于最新时间戳，跳出循环
                    if val_key <= current_latest_processed_ts:
                        # 将每次的idx赋予res_index，当循环结束时，res_index就是小于或等于最新时间戳的最大索引值
                        res_index = i
                        continue
                    else:
                        break
            # 将当前时间步的动态嵌入索引更新到dyemb_timestamp中
            dyemb_timestamp[:, t:t + step_len] = res_index

        # 创建一个列表，用于存储每个时间步的用户嵌入
        dyuser_emb_list = list()
        # dynamic_node_emb_dict's values are node embeddings (tensors)
        # These tensors should already be on self.device due to gnn_diffusion_layer (nn.Module) being on self.device
        for val_key in sorted(dynamic_node_emb_dict.keys()):
            # 使用embedding函数从输入序列中获取对应时间步的用户嵌入：
            # input_seq中的用户ID作为索引，从嵌入矩阵dynamic_node_emb_dict[val_key]中检索对应的嵌入向量
            # input_seq：[batch_size, seq_len]; dynamic_node_emb_dict[val_key]：[num_users, embedding_dim]
            # dyuser_emb_sub = F.embedding(input_seq.cuda(), dynamic_node_emb_dict[val_key].cuda()).unsqueeze(2)  # [batch_size, seq_len, 1, embedding_dim]
            # input_seq is already on self.device. dynamic_node_emb_dict[val_key] should also be.
            dyuser_emb_sub = F.embedding(input_seq, dynamic_node_emb_dict[val_key]).unsqueeze(
                2)  # [batch_size, seq_len, 1, embedding_dim] # MODIFIED: Removed .cuda()
            dyuser_emb_list.append(dyuser_emb_sub)  # [steps, batch_size, seq_len, 1, embedding_dim]

        # 将所有时间步的用户嵌入沿第二维拼接，形成完整的用户嵌入张量
        # 即将dyuser_emb_list里的每个张量[batch_size, seq_len, 1, embedding_dim]沿着第二维拼接起来，
        # 最终得到[batch_size, seq_len, num_steps, embedding_dim]
        if not dyuser_emb_list:  # ADDED: Handle empty list case
            # Fallback if no dynamic embeddings could be generated (e.g., empty dynamic_node_emb_dict)
            # This situation should ideally be avoided by ensuring dynamic_node_emb_dict is populated.
            # Create a zero tensor of the expected shape or handle error appropriately.
            # For now, creating zeros as a placeholder. This might lead to poor performance if hit.
            dyemb = torch.zeros(batch_size, max_len, self.embedding_size, device=self.device)
        else:
            dyuser_emb = torch.cat(dyuser_emb_list, dim=2)  # [bth, seq_len, time_step, hidden_size]

            # 通过时间注意力机制融合不同时间步的用户嵌入
            # dyemb = self.time_attention(dyemb_timestamp.cuda(), dyuser_emb.cuda())  # [bth, seq_len, hidden_size]
            # dyemb_timestamp and dyuser_emb are already on self.device
            dyemb = self.time_attention(dyemb_timestamp,
                                        dyuser_emb)  # [bth, seq_len, hidden_size] # MODIFIED: Removed .cuda()

        # 对动态嵌入应用dropout
        dyemb = self.dropout(dyemb)

        # 将动态嵌入和顺序嵌入沿最后一维拼接，形成最终的嵌入表示
        # final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()  # [bth, seq_len, hidden_size+pos_dim]
        final_embed = torch.cat([dyemb, order_embed],
                                dim=-1)  # [bth, seq_len, hidden_size+pos_dim] # MODIFIED: Removed .cuda()

        # 使用decoder_attention模块计算自注意力输出，使用掩码处理填充位置
        # att_out = self.decoder_attention(final_embed.cuda(),
        #                                  final_embed.cuda(),
        #                                  final_embed.cuda(),
        #                                  mask=mask.cuda())  # [batch_size, seq_len, hidden_size+pos_dim]
        # final_embed and mask are already on self.device
        att_out = self.decoder_attention(final_embed,
                                         final_embed,
                                         final_embed,
                                         mask=mask)  # [batch_size, seq_len, hidden_size+pos_dim] # MODIFIED: Removed .cuda()
        # 对注意力输出应用dropout
        # att_out = self.dropout(att_out.cuda())
        att_out = self.dropout(att_out)  # MODIFIED: Removed .cuda() (dropout input is already on device)

        # 通过线性层将注意力输出转换为最终的输出表示
        # output = self.linear(att_out.cuda())  # (batch_size, seq_len, |U|)
        output = self.linear(att_out)  # (batch_size, seq_len, |U|) # MODIFIED: Removed .cuda()

        # 获取之前用户的掩码，用于调整输出结果
        # mask_prev_user = self.get_previous_user_mask(input_seq.cuda(), self.user_num)
        mask_prev_user = self.get_previous_user_mask(input_seq,
                                                     self.user_num)  # input_seq is already on self.device # MODIFIED
        # 将掩码添加到输出中，进行适当的调整
        # output = output.cuda() + mask_prev_user.cuda()
        output = output + mask_prev_user  # MODIFIED: Both are on self.device

        # 将输出调整为 (batch_size * user_len, |U|) 的形状并返回
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

        # 获取 pred 中每行的最大值的索引，表示模型认为每个时间步最可能的类别,pred.max(1) 返回一个包含最大值和索引的元组，而 [1] 表示取出索引部分。
        pred_choice = pred.max(1)[1]  # pred_choice will be on self.device
        gold_flat = gold.contiguous().view(-1)  # 将 gold 转换为一维数组，确保它与 pred 的展平形状一致。 gold_flat on self.device
        n_correct = pred_choice.data.eq(
            gold_flat.data)  # 比较 pred 和 gold，返回一个布尔数组，表示每个位置是否预测正确。 n_correct on self.device
        # gold.ne(Constants.PAD): 生成一个布尔数组，标记 gold 中不是填充符的部分。
        mask_correct = gold_flat.ne(Constants.PAD).data  # mask_correct on self.device
        # masked_select(...): 只选择有效的（非填充）预测，确保不会将填充位置计入正确预测。
        # sum().float(): 最后计算正确预测的数量，并将其转换为浮点数。
        n_correct = n_correct.masked_select(mask_correct).sum().float()  # result is a scalar tensor on self.device
        return loss, n_correct
