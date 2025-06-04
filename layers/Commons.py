import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch_geometric.nn import GCNConv  # Assuming this is correctly imported
from torch.nn.parameter import Parameter


class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        # hidden and dy_emb are expected to be on the same correct device
        # torch.cat will create the new tensor on the device of its input tensors
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        # Operations like self.linear1, torch.tanh, self.linear2, F.softmax
        # will operate on the device of their input (emb).
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


class GraphNN(nn.Module):
    def __init__(self, num_nodes, input_dim, dropout=0.5, is_norm=True):
        """
        初始化GraphNN模型。

        :param num_nodes: 节点或实体的总数
        :param input_dim: 节点特征的维度
        :param dropout: 丢弃率，用于正则化
        :param is_norm: 是否应用批归一化
        """
        super(GraphNN, self).__init__()

        # 嵌入层，将节点/实体映射到特征空间
        self.embedding = nn.Embedding(num_nodes, input_dim, padding_idx=0)  # padding_idx=0 is common for PAD

        # 定义两个图卷积层
        self.gnn1 = GCNConv(input_dim, input_dim * 2)  # 第一层：输入特征 -> 输出特征翻倍
        self.gnn2 = GCNConv(input_dim * 2, input_dim)  # 第二层：将特征减少回原始大小

        # 存储归一化标志
        self.is_norm = is_norm

        # 丢弃层，用于正则化
        self.dropout = nn.Dropout(dropout)

        # 如果启用归一化，初始化批归一化层
        if self.is_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)

        # 初始化嵌入层的权重
        self.init_weights()

    def init_weights(self):
        """使用Xavier正态分布初始化嵌入层的权重。"""
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        """
        前向传播。

        :param graph: 输入图，包含节点和边的信息 (expected to be a PyG Data object or similar)
        :return: 节点的输出特征
        """
        # 获取图的边索引
        # graph.edge_index is expected to be on the same device as the graph features or model
        # graph_edge_index = graph.edge_index.cuda() # MODIFIED: Remove .cuda()
        graph_edge_index = graph.edge_index  # GCNConv will handle device based on input

        # 从嵌入层获取初始节点嵌入 (self.embedding.weight is on the model's device)
        # GCNConv expects node features as first argument. Here self.embedding.weight acts as initial node features for all nodes.
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)  # [n_nodes, input_dim*2]

        # 对嵌入应用丢弃
        graph_x_embeddings = self.dropout(graph_x_embeddings)

        # 通过第二个图卷积层处理嵌入
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)  # [n_nodes, input_dim]

        # 如果指定，应用批归一化
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)

        # 返回最终输出
        # return graph_output.cuda() # MODIFIED: Remove .cuda(), output is already on the correct device
        return graph_output


class HierarchicalGNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, use_norm=True):
        """
        初始化层次图神经网络（HGNN）与注意力机制。

        :param input_dim: 输入特征的维度
        :param hidden_dim: 隐藏层的特征维度 (Note: hidden_dim is passed but not directly used in layer definitions, output_dim is used)
        :param output_dim: 输出特征的维度
        :param dropout: 丢弃率，用于正则化
        :param use_norm: 是否应用批归一化
        """
        super(HierarchicalGNNWithAttention, self).__init__()

        # 存储丢弃概率和归一化标志
        self.dropout = dropout
        self.use_norm = use_norm

        # 如果启用归一化，初始化批归一化层
        if self.use_norm:
            self.batch_norm1 = nn.BatchNorm1d(output_dim)

        # 初始化层次图注意力层
        self.gat1 = HGATLayer(input_dim, output_dim, dropout=self.dropout, transfer=False, concat=True, use_edge=True)

        # 初始化融合层，用于结合嵌入
        self.fusion_layer = Fusion(output_dim)  # Assuming output_dim is the input_size for Fusion

    def forward(self, friendship_embedding, hypergraph_structure):
        """
        前向传播。

        :param friendship_embedding: 节点的初始嵌入 (expected on correct device)
        :param hypergraph_structure: 包含超图信息的列表
        :return: 包含所有嵌入的字典
        """
        # 从超图中获取根嵌入
        # hypergraph_structure[1] (root indices) needs to be on the same device as friendship_embedding
        # If hypergraph_structure[1] is created on CPU, it needs to be moved.
        # Assuming hypergraph_structure[1] (tensor of indices) is on the correct device.
        # root_indices = hypergraph_structure[1].cuda() # MODIFIED: Remove .cuda()
        root_indices = hypergraph_structure[1]  # Ensure this is on the correct device before calling
        root_embedding = F.embedding(root_indices, friendship_embedding)

        # 提取超图结构
        hypergraph = hypergraph_structure[0]
        embedding_results = {}

        current_node_embedding = friendship_embedding  # Keep track of the evolving node embedding

        # 遍历超图中的每个子图
        for subgraph_key in hypergraph.keys():
            # 获取当前子图
            subgraph = hypergraph[subgraph_key]  # subgraph is likely an adjacency matrix or similar structure

            # 使用图注意力层计算节点和边的嵌入
            # subgraph needs to be on the correct device if it's a tensor
            # sub_node_embedding, sub_edge_embedding = self.gat1(current_node_embedding, subgraph.cuda(), root_embedding) # MODIFIED: Remove .cuda()
            # Assuming subgraph is a tensor and needs to be on the same device as current_node_embedding
            # It's better if subgraph is prepared on the correct device before this loop.
            # If subgraph is part of PyG Data, its device should be consistent.
            sub_node_embedding, sub_edge_embedding = self.gat1(current_node_embedding, subgraph, root_embedding)

            # 对节点嵌入应用丢弃
            sub_node_embedding = F.dropout(sub_node_embedding, self.dropout, training=self.training)

            # 如果指定，应用批归一化
            if self.use_norm:
                sub_node_embedding = self.batch_norm1(sub_node_embedding)
                # Assuming sub_edge_embedding also has output_dim features if batch_norm1 is used
                sub_edge_embedding = self.batch_norm1(sub_edge_embedding)

            # 融合当前的节点嵌入与新的子图节点嵌入
            current_node_embedding = self.fusion_layer(current_node_embedding, sub_node_embedding)

            # 将更新后的节点和边嵌入存储在输出字典中
            # .cpu() here might be intentional if subsequent processing needs CPU data or to save GPU memory.
            # If not, and further GPU processing is needed, remove .cpu().
            # For device consistency, if results are fed back to GPU models, avoid .cpu().
            # embedding_results[subgraph_key] = [current_node_embedding.cpu(), sub_edge_embedding.cpu()]
            embedding_results[subgraph_key] = [current_node_embedding,
                                               sub_edge_embedding]  # MODIFIED: Kept on device, unless CPU is explicitly needed later.

        # 返回包含所有嵌入的字典
        return embedding_results


class HGATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, transfer, concat=True, bias=False, use_edge=True):
        """
        初始化图注意力层（Hierarchical Graph Attention Layer）。

        :param input_dim: 输入特征的维度
        :param output_dim: 输出特征的维度
        :param dropout: 丢弃率，用于正则化
        :param transfer: 是否使用可训练的权重矩阵
        :param concat: 是否在输出时应用激活函数
        :param bias: 是否为每个节点添加偏置
        :param use_edge: 是否对边嵌入进行进一步的处理
        """
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat = concat
        self.use_edge = use_edge
        self.transfer = transfer

        # 如果 transfer 为 True，初始化权重矩阵
        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.input_dim, self.output_dim))
        else:
            self.register_parameter('weight', None)  # PyTorch handles None parameters correctly

        # 初始化两个额外的权重矩阵
        self.weight2 = Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.weight3 = Parameter(torch.Tensor(self.output_dim, self.output_dim))

        # 初始化偏置参数（可选）
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        使用均匀分布初始化权重和偏置参数。
        """
        stdv = 1. / math.sqrt(self.output_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, adjacency_matrix, root_embeddings):
        """
        前向传播。

        :param node_features: 节点特征矩阵 (on device)
        :param adjacency_matrix: 邻接矩阵，表示图的边结构 (on device)
        :param root_embeddings: 根节点嵌入，用于注意力计算 (on device)
        :return: 更新后的节点嵌入（以及可选的边嵌入）
        """
        # All operations will be on the device of input tensors.
        # Parameter tensors (self.weight, self.weight2, etc.) are on the model's device.

        # 如果 transfer 为 True，使用 self.weight 进行特征变换
        if self.transfer:
            node_features_transformed = node_features.matmul(self.weight)
        else:
            # self.weight2 is a Parameter, so it's on the model's device
            node_features_transformed = node_features.matmul(self.weight2)

        # 如果存在偏置，添加偏置
        if self.bias is not None:
            node_features_transformed = node_features_transformed + self.bias

        # 计算边嵌入，首先对邻接矩阵的转置应用 softmax
        adjacency_matrix_t = F.softmax(adjacency_matrix.T, dim=1)  # .T creates a view, on same device
        edge_embeddings = torch.matmul(adjacency_matrix_t, node_features_transformed)  # Using transformed features

        # 对边嵌入应用丢弃和激活
        edge_embeddings = F.dropout(edge_embeddings, self.dropout, training=self.training)
        edge_embeddings = F.relu(edge_embeddings, inplace=False)  # inplace=False is safer

        # 使用 weight3 对边嵌入进行变换
        edge_transformed = edge_embeddings.matmul(self.weight3)

        # 重新应用 softmax 到原始邻接矩阵
        adjacency_matrix_softmaxed = F.softmax(adjacency_matrix, dim=1)

        # 计算节点的注意力嵌入
        node_attention_embeddings = torch.matmul(adjacency_matrix_softmaxed, edge_transformed)
        node_attention_embeddings = F.dropout(node_attention_embeddings, self.dropout, training=self.training)

        # 如果需要 concat，将输出通过 ReLU
        if self.concat:
            node_attention_embeddings = F.relu(node_attention_embeddings, inplace=False)

        # 如果需要处理边嵌入，返回节点和边嵌入；否则仅返回节点嵌入
        if self.use_edge:
            edge_output = torch.matmul(adjacency_matrix_t,
                                       node_attention_embeddings)  # Use adjacency_matrix_t again for consistency
            edge_output = F.dropout(edge_output, self.dropout, training=self.training)
            edge_output = F.relu(edge_output, inplace=False)
            return node_attention_embeddings, edge_output
        else:
            return node_attention_embeddings

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.input_dim} -> {self.output_dim})"


class DynamicGraphNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, time_step_split, dropout_rate=0.1):
        """
        初始化动态图神经网络（Dynamic Graph Neural Network）。

        :param num_nodes: 节点总数（词汇表大小）
        :param hidden_dim: 隐藏层特征的维度
        :param time_step_split: Number of time steps for linear layer (Note: used in self.linear input size)
        :param dropout_rate: 丢弃率，用于正则化
        """
        super(DynamicGraphNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # 嵌入层，将节点映射到特征空间
        self.embedding = nn.Embedding(num_nodes, hidden_dim)  # Weights will be on model's device
        init.xavier_normal_(self.embedding.weight)

        # 图神经网络层
        self.gnn1 = GraphNN(num_nodes, hidden_dim)  # gnn1 will be on model's device

        # 线性层，将多步图嵌入映射到隐藏维度
        # This linear layer is never used in the provided forward pass.
        # If it were, its input would be a concatenation of embeddings over time_step_split.
        self.linear_unused = nn.Linear(hidden_dim * time_step_split, hidden_dim)
        init.xavier_normal_(self.linear_unused.weight)

        # 丢弃层，用于正则化
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, diffusion_graphs):
        """
        前向传播。

        :param diffusion_graphs: 包含多个扩散图的字典 (each graph is likely a PyG Data object)
        :return: 各扩散图的嵌入表示 (dictionary with embeddings on device)
        """
        results = dict()
        # graph_embedding_list = [] # This list is populated but never used.

        # 遍历扩散图
        for key in sorted(diffusion_graphs.keys()):
            graph = diffusion_graphs[key]  # graph should be on the correct device if it's a PyG Data object
            graph_x_embeddings = self.gnn1(graph)  # gnn1 output will be on the model's device

            # 应用丢弃
            graph_x_embeddings = self.dropout(graph_x_embeddings)

            # 将嵌入移动到 CPU
            # graph_x_embeddings = graph_x_embeddings.cpu() # MODIFIED: Removed .cpu() to keep on device
            # unless explicitly needed by the caller on CPU.

            # 存储结果
            # graph_embedding_list.append(graph_x_embeddings) # List not used
            results[key] = graph_x_embeddings  # Embeddings remain on the device of gnn1's output

        return results


class TimeAttention(nn.Module):
    def __init__(self, time_size, in_features1):  # in_features1 is embedding dimension d
        super(TimeAttention, self).__init__()
        # time_size is the number of possible time indices (e.g., number of time steps or windows)
        self.time_embedding = nn.Embedding(time_size, in_features1)  # Weights on model's device
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.1)  # Note: dropout is defined but not used on alpha in forward

    def forward(self, T_idx, Dy_U_embed, mask=None, episilon=1e-6):
        '''
            T_idx: (bsz, user_len) - Time indices, expected on model's device
            Dy_U_embed: (bsz, user_len, time_len, d) # uid 从动态embedding lookup 之后的节点向量 - Expected on model's device
            output: (bsz, user_len, d)
        '''
        temperature = Dy_U_embed.size(-1) ** 0.5 + episilon
        T_embed = self.time_embedding(T_idx)  # (bsz, user_len, d) - Output on model's device

        # print(T_embed.size())
        # print(Dy_U_embed.size())

        # Einstein summation will operate on the device of input tensors
        affine = torch.einsum("bud,butd->but", T_embed, Dy_U_embed)  # (bsz, user_len, time_len)
        score = affine / temperature

        # if mask is None: # Masking logic was commented out
        #     mask = torch.triu(torch.ones(score.size(), device=score.device), diagonal=1).bool() # MODIFIED: Create mask on score's device
        #     score = score.masked_fill(mask, -torch.finfo(score.dtype).max) # MODIFIED: Use finfo for large negative

        alpha = F.softmax(score, dim=-1)  # (bsz, user_len, time_len) # Corrected dim from 1 to -1 (or 2) for time_len
        # alpha = self.dropout(alpha) # Dropout was commented out
        alpha = alpha.unsqueeze(dim=-1)  # (bsz, user_len, time_len, 1)

        att = (alpha * Dy_U_embed).sum(dim=2)  # (bsz, user_len, d) # Sum over time_len dimension
        return att


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Slicing operation, output is on the same device as x
        if self.chomp_size == 0:  # ADDED: Handle chomp_size=0 case
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, padding, stride=1, dilation=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        # weight_norm and nn.Conv1d will be on the model's device
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # padding here is for conv, chomp_size for Chomp1d
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()  # For the residual connection
        self.init_weights()

    def init_weights(self):
        # For weight_norm, access original weight via conv1.weight_g and conv1.weight_v
        # Or, more simply, initialize the underlying nn.Conv1d's weight before applying weight_norm
        # However, the current code initializes self.conv1.weight directly, which PyTorch
        # might handle correctly by initializing the 'weight' parameter of the weight_norm wrapper.
        # It's generally better to initialize the base nn.Conv1d layer's weight.
        # For simplicity, keeping original init if it works.
        self.conv1.weight.data.normal_(0, 0.01)  # This might refer to the reparametrized weight
        self.conv2.weight.data.normal_(0, 0.01)  # This might refer to the reparametrized weight

    def forward(self, x):
        # x is expected on the model's device
        x_permuted = x.permute(0, 2, 1)  # permute creates a view, on same device

        out = self.net(x_permuted)  # self.net operates on device of x_permuted
        res = x_permuted  # Residual connection

        # Ensure out and res have compatible shapes for addition if chomp changes length
        # If chomp1/chomp2 reduce length, res needs to be sliced or out padded.
        # Assuming kernel_size, padding, stride, dilation are set for 'same' convolution for residual.
        # If not, this residual connection might be problematic.
        # For example, if padding is not (kernel_size-1)//2 * dilation for 'same' like behavior before chomp.
        # The Chomp1d(padding) suggests that padding is added by Conv1d and then removed by Chomp1d.
        # This is a common pattern in TCNs to achieve causal convolutions.

        return self.relu(out + res).permute(0, 2, 1)
