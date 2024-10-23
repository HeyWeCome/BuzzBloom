import numpy as np
import torch
import pickle
from utils import Constants
import os
from torch_geometric.data import Data
import scipy.sparse as sp
import torch.nn.functional as F


def build_friendship_network(dataloader):
    # 初始化选项和用户索引字典
    # options = Options(dataloader)
    _u2idx = {}

    # 从文件中加载用户到索引的映射
    with open(dataloader.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    edges_list = []

    # 检查关系数据文件是否存在
    if os.path.exists(dataloader.net_data):
        # 读取关系数据
        with open(dataloader.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            # 将每条关系分割为用户对
            relation_list = [edge.split(',') for edge in relation_list]

            # 根据索引字典将用户ID转换为索引
            relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _u2idx and edge[1] in _u2idx]
            # 反转边并添加到边列表
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            edges_list += relation_list_reverse
    else:
        # 如果数据文件不存在，返回空列表
        return []

    # 将边列表转换为张量并初始化边权重
    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()

    # 创建图数据对象并返回
    friend_ship_network = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)
    return friend_ship_network


def build_diff_hyper_graph_list(cascades, timestamps, user_size, step_split=Constants.step_split):
    """构建扩散超图的列表"""

    # 从级联数据和时间戳构建扩散图
    times, root_list = build_hyper_diff_graph(cascades, timestamps, user_size)

    # 初始化零向量和单位向量
    zero_vec = torch.zeros_like(times)
    one_vec = torch.ones_like(times)

    time_sorted = []
    graph_list = {}

    # 将所有时间戳合并并排序
    for time in timestamps:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)

    # 计算每个子图的长度
    split_length = len(time_sorted) // step_split

    # 根据时间划分子图
    for x in range(split_length, split_length * step_split, split_length):
        if x == split_length:
            sub_graph = torch.where(times > 0, one_vec, zero_vec) - torch.where(times > time_sorted[x],
                                                                                one_vec,
                                                                                zero_vec)
        else:
            sub_graph = torch.where(times > time_sorted[x - split_length], one_vec, zero_vec) - torch.where(
                times > time_sorted[x], one_vec, zero_vec)

        graph_list[time_sorted[x]] = sub_graph

    # 返回包含子图和根用户列表的结果
    graphs = [graph_list, root_list]

    return graphs


def build_hyper_diff_graph(cascades, timestamps, user_size):
    """返回超图的邻接矩阵和时间邻接矩阵"""

    # 计算级联数量和用户数量
    e_size = len(cascades) + 1
    n_size = user_size
    rows = []
    cols = []
    vals_time = []
    root_list = [0]  # 根用户列表初始化

    # 遍历每个级联以构建邻接关系
    for i in range(e_size - 1):
        root_list.append(cascades[i][0])  # 记录根用户
        rows += cascades[i][:-1]  # 添加级联中的用户到行索引
        cols += [i + 1] * (len(cascades[i]) - 1)  # 添加列索引
        vals_time += timestamps[i][:-1]  # 收集时间戳

    # 转换根用户列表为张量
    root_list = torch.tensor(root_list)

    # 创建稀疏邻接矩阵
    Times = torch.sparse_coo_tensor(torch.Tensor([rows, cols]), torch.Tensor(vals_time), [n_size, e_size])

    # 返回稠密邻接矩阵和根用户列表
    return Times.to_dense(), root_list
