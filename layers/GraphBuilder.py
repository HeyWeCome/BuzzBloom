from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import pickle
from utils import Constants
import os
from torch_geometric.data import Data
import scipy.sparse as sp
import torch.nn.functional as F


def build_friendship_network(dataloader):
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


def build_dynamic_heterogeneous_graph(dataloader, time_step_split):
    # 初始化两个字典用于存储用户和索引的映射关系
    _u2idx = {}  # 用户到索引的映射
    _idx2u = []  # 索引到用户的映射

    # 从dataloader中加载用户到索引的映射字典
    with open(dataloader.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)  # 反序列化用户到索引的映射字典

    # 从dataloader中加载索引到用户的映射字典
    with open(dataloader.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)  # 反序列化索引到用户的映射字典

    follow_relation = []  # 用于存储关注关系的列表（有向关系）

    # 检查社交网络数据文件是否存在，friendship关系
    if os.path.exists(dataloader.net_data):
        with open(dataloader.net_data, 'r') as handle:
            # 读取网络数据中的所有边，并去除前后空白符
            edges_list = handle.read().strip().split("\n")
            edges_list = [edge.split(',') for edge in edges_list]  # 每条边通过逗号分隔成节点对
            # 将用户ID转换为索引，构建关注关系的元组列表
            follow_relation = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in edges_list if
                               edge[0] in _u2idx and edge[1] in _u2idx]

    # 加载动态扩散图（例如信息传播图），返回一个以时间步为键的字典，每个时间步对应一个图的邻接列表形式
    dy_diff_graph_list = load_dynamic_diffusion_graph(dataloader, time_step_split)

    dynamic_graph = dict()  # 初始化用于存储动态图的字典

    # 按时间步的顺序遍历扩散图
    for x in sorted(dy_diff_graph_list.keys()):
        # 初始化当前时间步的边列表，初始为关注关系
        edges_list = follow_relation
        edges_type_list = [0] * len(follow_relation)  # 关注关系的类型标记为0
        edges_weight = [1.0] * len(follow_relation)  # 关注关系的权重初始化为1.0

        # 将扩散图中的重发关系添加到当前时间步的边列表中
        for key, value in dy_diff_graph_list[x].items():
            edges_list.append(key)  # 将扩散关系的边添加到边列表
            edges_type_list.append(1)  # 重发关系的类型标记为1
            edges_weight.append(sum(value))  # 将权重计算为该重发关系的值之和

        # 将边列表转换为张量，并进行转置，使得其适合图数据结构的要求
        edges_list_tensor = torch.LongTensor(edges_list).t()
        # 转换边类型和权重为张量
        edges_type = torch.LongTensor(edges_type_list)
        edges_weight = torch.FloatTensor(edges_weight)

        # 使用PyTorch Geometric库构建图数据对象
        data = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_weight=edges_weight)
        # 将当前时间步的图数据添加到动态图中
        dynamic_graph[x] = data

    # 返回构建好的动态图字典
    return dynamic_graph


def load_dynamic_diffusion_graph(dataloader, time_step_split):
    """
    构建动态扩散图（例如信息传播图）。
    :param dataloader: 提供数据的加载器，包含用户映射和训练数据。
    :param time_step_split: 时间步的划分数量，用于确定时间步长。
    :return: 包含各个时间步的动态扩散图字典。
    """
    # 初始化两个字典，用于存储用户到索引的映射关系
    _u2idx = {}  # 用户到索引的映射
    _idx2u = []  # 索引到用户的映射

    # 从dataloader中加载用户到索引和索引到用户的映射字典
    with open(dataloader.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)  # 反序列化用户到索引的映射字典
    with open(dataloader.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)  # 反序列化索引到用户的映射字典

    # 从训练数据中提取级联信息及其对应的时间戳
    cascades, timestamps, _ = dataloader.train_data

    # 初始化一个列表，用于存储用户交互对及其时间戳
    t_cascades = []

    # 遍历每一个级联及其对应的时间戳
    for cascade, timestamp in zip(cascades, timestamps):
        # 为每个级联创建一个用户-时间戳对的列表
        userlist = list(zip(cascade, timestamp))

        # 创建连续用户对及其交互时间（例如，[用户A, 用户B, 时间]）
        """
        如果：userlist为[(用户A, 时间1), (用户B, 时间2), (用户C, 时间3)]
        那么 zip(userlist[:-1], userlist[1:]) 会配对出：
        [(用户A, 时间1), (用户B, 时间2)]
        [(用户B, 时间2), (用户C, 时间3)]
        pair_user 中保存的就是这些配对中的用户及其发生交互的时间，结果为：
        [(用户A, 用户B, 时间2), (用户B, 用户C, 时间3)]
        
        其中，用户A 和 用户B 之间的交互发生在 时间2，用户B 和 用户C 之间的交互发生在 时间3。
        这个操作的目的是基于时间顺序连接级联中的连续用户对，从而捕捉用户之间的信息传播过程。
        """
        pair_user = [(i[0], j[0], j[1]) for i, j in zip(userlist[:-1], userlist[1:])]

        # 仅考虑长度在2到500之间的级联（即至少有1对用户交互，最多500对）
        if len(pair_user) > 1 and len(pair_user) <= 500:
            t_cascades.extend(pair_user)  # 将符合条件的用户对添加到总的级联列表中

    # 将用户交互对列表转换为pandas DataFrame
    t_cascades_pd = pd.DataFrame(t_cascades, columns=["user1", "user2", "timestamp"])

    # 根据时间戳对用户交互进行排序
    t_cascades_pd = t_cascades_pd.sort_values(by="timestamp")

    # 计算交互对的总数量，并根据时间步划分数量计算每个时间步的长度
    t_cascades_length = t_cascades_pd.shape[0]  # 总交互数
    step_length_x = t_cascades_length // time_step_split  # 每个时间步的长度

    # 初始化字典，用于存储每个时间步的级联交互对
    t_cascades_list = dict()

    # 按照时间步划分数据，并提取对应的用户交互对
    for x in range(step_length_x, t_cascades_length - step_length_x, step_length_x):
        # 获取当前时间步的数据子集
        t_cascades_pd_sub = t_cascades_pd[:x]

        # 提取当前时间片中的用户对
        t_cascades_sub_list = t_cascades_pd_sub.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()

        # 获取该子集的最大时间戳，作为该时间步的标识
        sub_timesas = t_cascades_pd_sub["timestamp"].max()

        # 将用户对存储在对应的时间步中
        t_cascades_list[sub_timesas] = t_cascades_sub_list

    # 处理最后一个时间步（包括所有的交互对）
    t_cascades_sub_list = t_cascades_pd.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
    sub_timesas = t_cascades_pd["timestamp"].max()
    t_cascades_list[sub_timesas] = t_cascades_sub_list

    # 初始化一个字典，用于存储每个时间步的动态扩散图
    dynamic_graph_dict_list = dict()

    # 遍历所有的时间步，按时间戳顺序构建扩散图
    for key in sorted(t_cascades_list.keys()):
        edges_list = t_cascades_list[key]  # 当前时间步的用户对列表

        # 创建一个字典，用于记录用户对之间的交互（例如：[用户A, 用户B]）
        cascade_dic = defaultdict(list)
        for upair in edges_list:
            cascade_dic[upair].append(1)  # 将交互记为1（可能表示一次交互）

        # 存储当前时间步的扩散图
        dynamic_graph_dict_list[key] = cascade_dic

    # 返回构建好的动态扩散图字典，其中每个时间步都对应一个扩散图
    return dynamic_graph_dict_list
