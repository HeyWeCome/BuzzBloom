import os
import pickle

import math
import numpy as np

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
import torch.nn.init as init
from torch_scatter import scatter_add
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

from utils import Constants

# This is a workaround for a known issue with some MKL/OpenMP library versions on certain OSes.
# It prevents a crash that can occur when multiple libraries are loaded.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""
Disentangling Inter- and Intra-Cascades Dynamics for Information Diffusion Prediction
IEEE Transactions on Knowledge and Data Engineering (TKDE), 2025
"""


def csr_to_geometric(mm):
    # .tocoo() 将稀疏矩阵转为坐标格式（row, col, data）
    coo = mm.tocoo().astype(np.float32)
    # PyG 的 edge_index 格式为 [2, num_edges]，第一行是源节点，第二行是目标节点
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([col, row])
    # data为edge_attr 存储每条边权重
    data = torch.FloatTensor(coo.data)
    return Data(edge_index=index, edge_attr=data)


def get_previous_user_mask(seq, user_size):
    """
    为序列中的每个位置，生成一个掩码，用于屏蔽掉在该位置之前（包括该位置）已经出现过的所有用户。
    - seq: 输入的用户索引序列, 形状 (B, L), B是batch_size, L是序列长度。
    - user_size: 用户总数 N。
    - 返回: 一个掩码矩阵, 形状 (B, L, N)。
    """
    assert seq.dim() == 2

    # --- 步骤 1: 构建一个下三角矩阵来收集历史用户 ---
    # 准备一个形状为 (B, L, L) 的容器
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))

    # a. 将原始序列 seq 复制 L 次，得到一个 (B, L, L) 的张量
    # 假设 seq[0] = [u1, u2, u3]
    # seqs[0] 会变成:
    # [[u1, u2, u3],
    #  [u1, u2, u3],
    #  [u1, u2, u3]]
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))

    # b. 创建一个下三角掩码矩阵
    # previous_mask 是一个 L x L 的下三角矩阵 (包括对角线)，元素为1，其余为0。
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    previous_mask = np.tril(np.ones(prev_shape), k=0).astype('float32')  # k默认为0包含对角线
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()

    # c. 将下三角掩码应用到复制的序列上
    # masked_seq[b, i, :] 将会包含 seq[b, 0...i] 的用户，其余位置为0
    # 假设 seq[0] = [u1, u2, u3]
    # masked_seq[0] 会变成:
    # [[u1, 0,  0 ],  <- 在第0个时间步，只看得到 u1
    #  [u1, u2, 0 ],  <- 在第1个时间步，看得到 u1, u2
    #  [u1, u2, u3]]  <- 在第2个时间步，看得到 u1, u2, u3
    masked_seq = previous_mask * seqs.data.float()


    # --- 步骤 2: 将历史用户索引映射到最终的掩码矩阵 ---
    # a. 增加一个维度，用于处理 PAD (索引为0) 的情况，防止 scatter_ 出错
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    # masked_seq 形状变为 (B, L, L+1)
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)

    # b. 创建一个最终的掩码矩阵容器，初始化为0
    # ans_tmp 形状 (B, L, N)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()

    # c. 使用 scatter_ 核心操作
    # 这是最关键的一步。它将 masked_seq 中的用户索引作为“地址”，
    # 在 ans_tmp 中对应的地址上填入一个很大的负数。
    # ans_tmp.scatter_(dim, index, src)
    # - dim=2: 沿着用户维度进行操作
    # - index=masked_seq.long(): 指定要在哪个索引位置上填充值
    # - src=float(-1000): 要填充的值
    # 效果: 对于 batch b, 时间步 i，如果用户 u_j 在 masked_seq[b, i, :] 中出现过，
    #       那么 ans_tmp[b, i, u_j] 就会被设置为 -1000。
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    # 将其包装成 Variable (在较早的PyTorch版本中需要，现在可以省略)
    # masked_seq = Variable(masked_seq, requires_grad=False)

    return masked_seq.cuda()


class Hyperedge(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        self.heads = 1
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Tensor = None) -> Tensor:

        self.out_channels = x.size(-1)

        num_nodes, num_edges = x.size(0), 0
        alpha = None

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))

        out = out.view(-1, self.heads * self.out_channels)

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:

        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels // heads

        self.hyperedge_func = Hyperedge()

        # attention
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * self.out_channels, bias=False)
        self.lin2 = Linear(in_channels, heads * self.out_channels, bias=False)
        self.lin3 = Linear(in_channels, heads * self.out_channels, bias=False)

        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))
        self.att2 = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)

    def FFN(self, X):
        output = self.FFN_2(F.relu(self.FFN_1(X)))
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Tensor = None,
                hyperedge_attr: Tensor = None) -> Tensor:

        num_nodes, num_edges = x.size(0), 0

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        hyperedge_attr = self.hyperedge_func(x, hyperedge_index)

        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)
        hyperedge_attr = self.lin2(hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
        x_i = x[hyperedge_index[0]]
        x_j = hyperedge_attr[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))

        hyperedge_attr = out.view(-1, self.heads * self.out_channels)
        hyperedge_attr = self.lin3(hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
        x_i = x[hyperedge_index[0]]
        x_j = hyperedge_attr[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att2).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D, alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)

        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.1, reverse=False):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.reverse = reverse

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()


    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.gelu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            if self.reverse:
                mask = torch.tril(torch.ones(pad_mask.size()), diagonal=-1).bool().cuda()
            else:
                mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)

        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score_new = self.dropout(Q_K_score)
        V_att = Q_K_score_new.bmm(V)
        return V_att, Q_K_score


    def multi_head_attention(self, Q, K, V, mask):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)
            mask = mask.reshape(-1, mask.size(-1))

        V_att, att_score = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output, att_score


    def forward(self, Q, K, V, mask=None):
        V_att, kl_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output, kl_att

class MIM(nn.Module):

    @staticmethod
    def parse_model_args(parser):
        """
        Adds model-specific arguments to the argument parser.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which to add arguments.

        Returns:
            argparse.ArgumentParser: The parser with added arguments.
        """
        parser.add_argument('--att_head', type=int, default=8,
                            help='The number of attention heads.')
        parser.add_argument('--window', type=int, default=5,
                            help='Number of windows size.')
        parser.add_argument('--graph_layer', type=int, default=1,
                            help='The number of graph layers.')
        parser.add_argument('--beta', type=float, default=1.0,
                            help='IDP task maginitude.')
        parser.add_argument('--beta2', type=float, default=0.8,
                            help='KL task maginitude.')
        parser.add_argument('--beta3', type=float, default=0.2,
                            help='ssl task maginitude.')
        return parser

    def __init__(self, args, data_loader):
        super(MIM, self).__init__()
        self.args = args
        self.beta = args.beta
        self.beta2 = args.beta2
        self.beta3 = args.beta3

        self.device = args.device
        self.drop_r = args.dropout
        self.layers = args.graph_layer  # HGAT的层数 L
        self.att_head = args.att_head  # 注意力头的数量 Q
        self.win = args.window  # 依赖超图的级联窗口

        self.n_node = data_loader.user_num    # 用户总数 N
        self.all_cascades = data_loader.all_cascades    # test和valid的序列数据，供依赖超图和偏好超图的构建使用
        self.hidden_size = args.d_model  # 模型的核心维度 d

        # Multi-channel hypergraph construction
        # - H_S 社交超图 user_j与user_j是否存在三角闭包关系
        # - H_D 依赖超图 user_j是否是user_i的共现列表中
        # - H_P 偏好超图 user_j是否存在cas_i中
        self.graph = self.build_social_hypergraph(data_loader)
        self.H_Item, self.H_User = self.build_cas_hypergraph(self.n_node,self.all_cascades,self.win)
        self.n_channel = 3

        # Initial user embedding layer
        self.user_embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)

        # --- Multi-Intent Learning component ---
        # Self-Gating
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.hidden_size)) for _ in range(self.n_channel)])
        # Channel attention
        self.att = nn.Parameter(torch.zeros(1, self.hidden_size))
        self.att_m = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        # HGAT
        self.HGAT_layers = nn.ModuleList()
        for i in range(self.layers):
            self.HGAT_layers.append(
                HypergraphConv(in_channels=self.hidden_size, out_channels=self.hidden_size, heads=self.att_head))

        # --- Dual Temporal Dependency Modeling component ---
        # Past Encoder
        self.history_ATT = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head,
                                            attn_dropout=self.drop_r)
        # Future Encoder
        self.future_ATT = TransformerBlock(input_size=self.hidden_size, n_heads=self.att_head, attn_dropout=self.drop_r,
                                           reverse=True)

        self.reset_parameters()
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @staticmethod
    def build_social_hypergraph(dataloader):
        """
        构建社交感知超图 H_s = (A·A) ⊙ A：
            三角闭包思想，筛选出（你关注我，我关注他，你也关注他）这样的二度连接边
        """
        _u2idx = {}

        # 1. 加载用户-索引映射字典 {'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6}
        with open(dataloader.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)

        # 2. 读取社交网络
        edges_list = []
        # 检查关系数据文件是否存在
        if os.path.exists(dataloader.net_data):
            # 读取关系数据
            with open(dataloader.net_data, 'r') as handle:
                relation_list = handle.read().strip().split("\n")
                # 将每条关系分割为用户对 (A,B), (B,C), (A,C), (D,E)
                relation_list = [edge.split(',') for edge in relation_list]

                # 根据索引字典将用户ID转换为索引
                # # (A,B), (B,C), (A,C), (D,E) --> (2,3), (3,4), (2,4), (5,6)
                relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                                 edge[0] in _u2idx and edge[1] in _u2idx]
                # 反转边并添加到边列表
                relation_list_reverse = [edge[::-1] for edge in relation_list]
                edges_list += relation_list_reverse
        else:
            # 如果数据文件不存在，返回空列表
            return []

        # 3. 构建稀疏的邻接矩阵 A (Adjacency Matrix)
        row, col, entries = [], [], []
        for pair in edges_list:
            row += [pair[0]]
            col += [pair[1]]
            entries += [1.0]  # 边的权重为1.0
        social_mat = sp.csr_matrix((entries, (row, col)), shape=(len(_u2idx), len(_u2idx)), dtype=np.float32)

        # 4. 实现三角闭包公式: H_s = (A * A) ⊙ A
        social_matrix = social_mat.dot(social_mat)
        # ss.eye(...) ==> 添加自环，GNN中的常见做法
        social_matrix = social_matrix.multiply(social_mat) + sp.eye(len(_u2idx), dtype=np.float32)

        # 5. 将稀疏矩阵转换为PyTorch Geometric的Data对象
        social_matrix = social_matrix.tocoo()
        social_matrix = csr_to_geometric(social_matrix)

        return social_matrix

    @staticmethod
    def build_cas_hypergraph(user_size, all_cascade, win):
        """
        构建依赖感知超图 H_D：
            参与了同一个级联（转发了同一个信息）的用户群体，被认为有共同的偏好，因此每个级联构成一条超边。
        偏好感知超图 H_P：
            在一个级联传播序列中，通过一个滑动窗口，窗口内同时出现的用户被认为有扩散依赖关系，每个窗口内的用户集合构成一条超边。
            【注意】代码实现上对每个用户聚合了其所有相关的窗口，形成以用户为中心的超边
        """
        # ===================== 依赖感知超图 H_D (HG_User) 的构建 =====================
        # 核心思想：使用滑动窗口捕捉局部依赖关系
        """
        初始化 user_cont: user_cont = {2:[], 3:[], 4:[], 5:[], 6:[]}
        """
        user_cont = {}  # {key: user_idx, value: 与该user在同一个窗口内出现过的所有user的集合}
        for i in range(user_size):
            user_cont[i] = []

        for i in range(len(all_cascade)):
            """ 滑动窗口示例
            cas1：[2, 3, 5]
                窗口1：[2,3] 
                    user_cont更新：user_cont[2]->{2,3}, user_cont[3]->{2,3}
                窗口2：[3,5]
                    user_cont更新：user_cont[3]->{2,3,5}, user_cont[5]->{3,5}
            """
            cas = all_cascade[i]
            # 如果级联本身比窗口还短，则整个级联视为一个窗口，直接更新这个窗口内每个用户的共现列表
            if len(cas) < win:
                for idx in cas:
                    user_cont[idx] = list(set(user_cont[idx] + cas))
                continue
            # 滑动窗口遍历级联
            for j in range(len(cas) - win + 1):
                if (j + win) > len(cas):
                    break
                cas_win = cas[j:j + win]  # 取出一个窗口
                # 更新窗口内每个用户的共现列表
                for idx in cas_win:
                    user_cont[idx] = list(set(user_cont[idx] + cas_win))

        # 将 user_cont 字典构建成超图的关联矩阵 (Incidence Matrix)
        """
        H_U[i, j] = 1 表示用户 j 存在于用户 i 定义的超边中:
                     用户2(A) 用户3(B) 用户4(C) 用户5(D) 用户6(E)
        用户A的超边 [    1,      1,      1,      0,      0   ]
        用户B的超边 [    1,      1,      0,      1,      0   ]
        用户C的超边 [    1,      0,      1,      0,      1   ]
        用户D的超边 [    0,      1,      0,      1,      0   ]
        用户E的超边 [    0,      0,      1,      0,      1   ]
        """
        # 这里用CSR格式高效构建。indptr, indices, data是CSR的三要素
        indptr, indices, data = [], [], []
        indptr.append(0)
        idx = 0

        for j in user_cont.keys():
            # 累计没有共现用户的user个数，在最后构建关联矩阵时删除这些数量
            if len(user_cont[j]) == 0:
                idx = idx + 1
                continue
            # user_cont的每个key（一个用户）定义了一条超边，其value（共现用户列表）是这条超边的节点
            source = np.unique(user_cont[j])

            length = len(source)  # 超边节点个数
            s = indptr[-1]  # 最后一个指针位置
            indptr.append((s + length))  # 指针位置移动到当前用户超边尾端
            for i in range(length):
                indices.append(source[i])  # 将当前用户的超边节点添加到索引列表中
                data.append(1)  # 超边每个节点权重记为1

        # H_U 存储了 H_D 的关联矩阵
        H_U = sp.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys()) - idx, user_size))
        HG_User = H_U.tocoo()

        # ===================== 偏好感知超图 H_P (HG_Item) 的构建 =====================
        # 核心思想：每个完整的级联定义了一条超边
        """
            cas1: A, B, D (用户索引: [2, 3, 5])
            cas2: A, C, E (用户索引: [2, 4, 6])
            cas3: B, D (用户索引: [3, 5])

            H_T[i, j] = 1 表示用户 j 参与了超边 i:
                       用户2(A) 用户3(B) 用户4(C) 用户5(D) 用户6(E)
            超边1(cas1) [   1,      1,      0,      1,      0   ]
            超边2(cas2) [   1,      0,      1,      0,      1   ]
            超边3(cas3) [   0,      1,      0,      1,      0   ]
        """
        indptr, indices, data = [], [], []
        indptr.append(0)
        for j in range(len(all_cascade)):
            # 一条超边就是参与一个级联的所有独特用户
            items = np.unique(all_cascade[j])

            length = len(items)
            s = indptr[-1]
            indptr.append((s + length))
            for i in range(length):
                indices.append(items[i])
                data.append(1)

        # H_T 存储了 H_P 的关联矩阵，行是超边(级联)，列是节点(用户)
        H_T = sp.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))
        HG_Item = H_T.tocoo()

        # 4. 将两个超图的稀疏矩阵都转换为PyG的Data对象
        HG_Item = csr_to_geometric(HG_Item)
        HG_User = csr_to_geometric(HG_User)

        return HG_Item, HG_User

    def self_gating(self, em, channel):
        """
        为每个通道生成专属的用户特征
        @em 输入的嵌入矩阵，这里是 self.user_embedding.weight，形状 (N, d)
        @channel: 通道索引 (0, 1, or 2)
        """
        # 1. 线性变换: em @ self.weights[channel] (d, d) + self.bias[channel] -> (N, d)
        # 2. Sigmoid激活: torch.sigmoid(...) 即门控gate作用，得到每个值都在0到1之间 -> (N, d)
        # 3. 元素级乘法 (Hadamard Product): em * gate，实现了对原始嵌入的动态调整，得到了该通道专属的嵌入 -> (N, d)s
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def channel_attention(self, *channel_embeddings):
        """
        将 三个通道最终的用户表示矩阵 融合成一个用户表示：
            学习一个动态的权重，来表示这三个通道在当前情境下的相对重要性，然后进行加权求和
        @channel_embeddings: 一个包含多个通道嵌入矩阵的元组
                            (u_emb_c1, u_emb_c2, u_emb_c3)，每个形状都是 (N, d)
        """
        weights = []
        # --- 步骤 1: 为每个通道计算注意力分数 ---
        for embedding in channel_embeddings:
            # a. 线性变换: transformed_emb = torch.matmul(embedding, self.att_m) -> (N, d)
            #    - self.att_m: 一个可学习的共享权重矩阵 W_att, 形状 (d, d)

            # b. 与注意力向量相乘: torch.multiply(self.att, transformed_emb) -> (N, d)
            #    - self.att: 一个可学习的注意力查询向量 w^T, 形状 (1, d)
            #    - 利用广播机制，相当于将每个变换后的用户向量与查询向量w进行元素乘。

            # c. 求和得到分数: torch.sum(..., 1) -> (N,)
            #    - 沿着维度1（特征维度d）求和。
            #    - 这相当于计算了变换后的用户向量与查询向量w的点积
            #    - 结果是一个向量，形状 (N,)，每个值代表该用户在该通道下的“重要性分数”
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        # 此时，weights = [score_vec_c1, score_vec_c2, score_vec_c3]
        # 每个 score_vec_ci 的形状都是 (N,)

        # --- 步骤 2: 规范化分数 (Softmax) ---
        # a. 堆叠: torch.stack(weights, dim=0)
        #    - 将三个分数向量堆叠成一个矩阵, 形状 (3, N) (3个通道, N个用户)
        embs = torch.stack(weights, dim=0)

        # b. 转置并应用Softmax: F.softmax(embs.t(), dim=-1)
        #    - embs.t() 转置为 (N, 3)
        #    - F.softmax(..., dim=-1) 沿着最后一个维度（通道维度）进行softmax。
        #    - 这样，对于每个用户，其在三个通道上的分数就被归一化为和为1的概率分布。
        #    - score 形状: (N, 3). score[i] = [eta_S, eta_D, eta_P] for user i. 对应论文公式(9)的eta。
        score = F.softmax(embs.t(), dim=-1)

        # --- 步骤 3: 加权融合 ---
        mixed_embeddings = 0
        # 遍历每个通道
        for i in range(len(weights)):
            # a. 获取该通道的注意力权重: score.t()[i]
            #    - score.t() 转回 (3, N)
            #    - score.t()[i] 是一个行向量，包含了所有N个用户在该通道i上的权重。形状 (N,)

            # b. 获取该通道的用户嵌入: channel_embeddings[i]
            #    - 形状 (N, d)

            # c. 加权: 将每个用户的嵌入向量，乘以它对应的该通道的权重 -> (N, d)
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()

        # mixed_embeddings 是最终融合后的用户表示矩阵，对应论文公式(10)的 P -> (N, d)
        return mixed_embeddings, score

    def _dropout_graph(self, graph, keep_prob):
        edge_attr = graph.edge_attr
        edge_index = graph.edge_index.t()

        random_index = torch.rand(edge_attr.shape[0]) + keep_prob
        random_index = random_index.int().bool()

        edge_index = edge_index[random_index]
        edge_attr = edge_attr[random_index]
        return Data(edge_index=edge_index.t(), edge_attr=edge_attr)

    def history_cas_learning(self):
        """
        多通道意图学习流程
        """
        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1 - self.drop_r)
            H_User = self._dropout_graph(self.H_User, keep_prob=1 - self.drop_r)
            Graph = self._dropout_graph(self.graph, keep_prob=1 - self.drop_r)
        else:
            H_Item = self.H_Item
            H_User = self.H_User
            Graph = self.graph

        H_Item = H_Item.to(self.device)
        H_User = H_User.to(self.device)
        Graph = Graph.to(self.device)

        # 1. 通道专属特征初始化 (Self-Gating, 对应公式4)，得到特定通道的用户表示
        u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)  # 社交通道
        u_emb_c2 = self.self_gating(self.user_embedding.weight, 1)  # 偏好通道
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 2)  # 依赖通道
        all_emb_c1 = [u_emb_c1]
        all_emb_c2 = [u_emb_c2]
        all_emb_c3 = [u_emb_c3]

        # 2. 多层超图注意力传播 (对应公式5,6,7)
        for k in range(self.layers):
            # 关键：调用 HGAT 层进行图卷积
            # 在社交图上传播：
            u_emb_c1 = self.HGAT_layers[k](u_emb_c1, Graph.edge_index)
            normalize_c1 = F.normalize(u_emb_c1, p=2, dim=1)  # L2归一化
            all_emb_c1 += [normalize_c1]  # 存储每一层的输出，用于后续的层融合

            # 在偏好超图上传播：
            u_emb_c2 = self.HGAT_layers[k](u_emb_c2, H_Item.edge_index)
            normalize_c2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [normalize_c2]

            # 在依赖超图上传播：
            u_emb_c3 = self.HGAT_layers[k](u_emb_c3, H_User.edge_index)
            normalize_c3 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [normalize_c3]

        # 3. 层融合 (Layer Fusion, 对应公式8)
        # 将每一层的输出取平均，防止过平滑
        u_emb_c1 = torch.stack(all_emb_c1, dim=1)
        u_emb_c1 = torch.mean(u_emb_c1, dim=1)
        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.mean(u_emb_c2, dim=1)
        u_emb_c3 = torch.stack(all_emb_c3, dim=1)
        u_emb_c3 = torch.mean(u_emb_c3, dim=1)

        # 4. 多意图融合 (Channel Attention, 对应公式9,10)
        # high_embs 就是最终的用户意图表示矩阵 P
        high_embs, _ = self.channel_attention(u_emb_c1, u_emb_c2, u_emb_c3)
        return high_embs

    def forward(self, input_seq, input_timestamp=None, tgt_idx=None, label=None):
        if self.training:
            return self.forward_train(input_seq, label)

        # Move input to the correct device
        input_seq = input_seq.to(self.device)
        input = input_seq[:, :-1]

        mask = (input == Constants.PAD)

        HG_user = self.history_cas_learning()

        diff_emb = F.embedding(input, HG_user)
        past_att_out, past_dist = self.history_ATT(diff_emb, diff_emb, diff_emb, mask=mask.to(self.device))
        past_output = torch.matmul(past_att_out, torch.transpose(HG_user, 1, 0))

        mask = get_previous_user_mask(input.to(self.device), self.n_node)
        output_past = (past_output + mask).view(-1, past_output.size(-1))

        return output_past

    def forward_train(self, input_original, label):
        input = input_original
        mask = (input == Constants.PAD)
        mask_label = (label == Constants.PAD)

        # Get final user intent embedding -> (N, d)
        HG_user = self.history_cas_learning()

        # --- Dual Temporal Dependency Modeling ---
        # Prepare seq look-up -> (B, L_in, d)
        diff_emb = F.embedding(input, HG_user)
        future_emb = F.embedding(label, HG_user)

        # Past Encoding
        past_att_out, past_dist = self.history_ATT(diff_emb, diff_emb, diff_emb, mask=mask.to(self.device))
        # Future Encoding
        future_att_out, futrue_dist = self.future_ATT(future_emb, future_emb, future_emb, mask=mask_label.to(self.device))

        # Predicted Probability
        # Mapping outputs to user embedding space
        past_output = torch.matmul(past_att_out, torch.transpose(HG_user, 1, 0))
        future_output = torch.matmul(future_att_out, torch.transpose(HG_user, 1, 0))

        # block previous user
        mask = get_previous_user_mask(input.to(self.device), self.n_node)
        output_past = (past_output + mask).view(-1, past_output.size(-1))
        future_output = future_output.view(-1, past_output.size(-1))

        return output_past, future_output, past_dist, futrue_dist, F.normalize(past_att_out, p=2, dim=1), F.normalize(future_att_out, p=2, dim=1)

    def compute_kl(self, p, q):

        p_loss = F.kl_div(
            F.log_softmax(p + 1e-8, dim=-1), F.softmax(q + 1e-8, dim=-1), reduction="sum"
        )
        q_loss = F.kl_div(
            F.log_softmax(q + 1e-8, dim=-1), F.softmax(p + 1e-8, dim=-1), reduction="sum"
        )

        loss = (p_loss + q_loss) / 2
        return loss

    def kl_loss(self, attn, attn_reversed):

        loss = (self.compute_kl(attn.sum(dim=1).view(-1, self.att_head),
                                attn_reversed.sum(dim=1).view(-1, self.att_head)) +
                self.compute_kl(attn.sum(dim=2).view(-1, self.att_head),
                                attn_reversed.sum(dim=2).view(-1, self.att_head))) / 2
        return loss

    def seq2seqloss(self, inp_subseq_encodings: torch.Tensor,
                    label_subseq_encodings: torch.Tensor, input_cas: torch.Tensor) -> torch.Tensor:

        sqrt_hidden_size = np.sqrt(self.hidden_size)
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)
        normalized_dot_product = torch.sum(product, dim=-1) / sqrt_hidden_size
        numerator = torch.exp(normalized_dot_product)
        inp_subseq_encodings_trans_expanded = inp_subseq_encodings.unsqueeze(1)
        label_subseq_encodings_trans = label_subseq_encodings.transpose(1, 2)
        dot_products = torch.matmul(inp_subseq_encodings_trans_expanded, label_subseq_encodings_trans)
        dot_products = torch.exp(dot_products / sqrt_hidden_size)
        dot_products = dot_products.sum(-1)
        denominator = dot_products.sum(1)
        seq2seq_loss_k = -torch.log2(numerator / denominator)
        seq2seq_loss_k = torch.flatten(seq2seq_loss_k)
        input_cas = torch.flatten(input_cas)
        mask = (input_cas != Constants.PAD)
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, mask)
        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)
        return seq2seq_loss

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        """
        Calculates performance metrics during training, including the combined loss and prediction accuracy.

        Args:
            input_seq (torch.Tensor): The input sequence of user IDs.
            input_seq_timestamp (torch.Tensor): Timestamps for the input sequence.
            history_seq_idx (torch.Tensor): The cascade indices for the batch.
            loss_func (callable): The main prediction loss function (e.g., CrossEntropyLoss).
            gold (torch.Tensor): The ground truth next-user labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The total combined loss.
                - The number of correct predictions in the batch.
        """
        gold = gold.to(self.device)
        input_seq = input_seq.to(self.device)
        input_seq = input_seq[:, :-1]

        output_past, future_output, past_dist, futrue_dist, past_emb, future_emb = self.forward(
            input_seq=input_seq, label=gold
        )

        # 1. Main task loss (Prediction)
        loss = self.loss_function(output_past, gold.contiguous().view(-1))

        # Calculate accuracy
        pred_labels = output_past.max(1)[1]
        gold_flat = gold.contiguous().view(-1)
        # Only consider non-padded elements for accuracy calculation
        non_pad_mask = gold_flat.ne(Constants.PAD).data
        n_correct = pred_labels.data.eq(gold_flat.data).masked_select(non_pad_mask).sum().float()

        # 2. future_loss
        future_loss = loss_func(future_output, gold.contiguous().view(-1))

        # 3. kl_loss
        loss2 = self.kl_loss(past_dist, futrue_dist)

        # 4. seq2seqloss
        loss3 = self.seq2seqloss(past_emb, future_emb, input_seq)

        # Combine all losses
        total_loss = loss + self.beta* future_loss + self.beta2 * loss2 + self.beta3 * loss3
        return total_loss, n_correct
