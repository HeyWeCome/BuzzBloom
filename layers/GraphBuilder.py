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
    _u2idx = {}
    _idx2u = []

    with open(dataloader.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(dataloader.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    follow_relation = []  # directed relation
    if os.path.exists(dataloader.net_data):
        with open(dataloader.net_data, 'r') as handle:
            edges_list = handle.read().strip().split("\n")
            edges_list = [edge.split(',') for edge in edges_list]
            follow_relation = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in edges_list if
                               edge[0] in _u2idx and edge[1] in _u2idx]

    dy_diff_graph_list = load_dynamic_diffusion_graph(dataloader, time_step_split)
    dynamic_graph = dict()
    for x in sorted(dy_diff_graph_list.keys()):
        edges_list = follow_relation
        edges_type_list = [0] * len(follow_relation)  # 0:follow relation,  1:repost relation
        edges_weight = [1.0] * len(follow_relation)
        for key, value in dy_diff_graph_list[x].items():
            edges_list.append(key)
            edges_type_list.append(1)
            edges_weight.append(sum(value))

        edges_list_tensor = torch.LongTensor(edges_list).t()
        edges_type = torch.LongTensor(edges_type_list)
        edges_weight = torch.FloatTensor(edges_weight)

        data = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_weight=edges_weight)
        dynamic_graph[x] = data
    return dynamic_graph


def load_dynamic_diffusion_graph(dataloader, time_step_split):
    """
    Notice: we remove the code that repeated the construction of diffusion graphs with a different list-based format.
    :param dataloader:
    :param time_step_split:
    :return:
    """
    # Initialize dictionaries for user-to-index and index-to-user mappings
    _u2idx = {}
    _idx2u = []

    # Load pre-existing user-to-index and index-to-user mappings from pickled files
    with open(dataloader.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(dataloader.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    # Extract cascades and their corresponding timestamps from training data
    cascades, timestamps, _ = dataloader.train_data

    # Initialize an empty list to store user interaction pairs with timestamps
    t_cascades = []

    # Iterate over each cascade and its corresponding timestamps
    for cascade, timestamp in zip(cascades, timestamps):
        # Create a list of user-timestamp pairs for each cascade
        userlist = list(zip(cascade, timestamp))

        # Create pairs of consecutive users along with the interaction time
        pair_user = [(i[0], j[0], j[1]) for i, j in zip(userlist[:-1], userlist[1:])]

        # Consider only cascades with more than 1 pair and fewer than or equal to 500 pairs
        # Keep the original Setting.
        if len(pair_user) > 1 and len(pair_user) <= 500:
            t_cascades.extend(pair_user)

    # Convert the list of user interaction pairs into a pandas DataFrame
    t_cascades_pd = pd.DataFrame(t_cascades, columns=["user1", "user2", "timestamp"])

    # Sort the interactions based on timestamp
    t_cascades_pd = t_cascades_pd.sort_values(by="timestamp")

    # Calculate the total number of interactions and the length of each time step
    t_cascades_length = t_cascades_pd.shape[0]
    step_length_x = t_cascades_length // time_step_split

    # Dictionary to store cascades in different time steps
    t_cascades_list = dict()

    # Slice the data into time steps and extract the relevant user interactions
    for x in range(step_length_x, t_cascades_length - step_length_x, step_length_x):
        # Subset of data up to the current time step
        t_cascades_pd_sub = t_cascades_pd[:x]

        # Extract pairs of users for the current time slice
        t_cascades_sub_list = t_cascades_pd_sub.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()

        # Get the maximum timestamp for this subset
        sub_timesas = t_cascades_pd_sub["timestamp"].max()

        # Store the user interaction pairs under the corresponding timestamp
        t_cascades_list[sub_timesas] = t_cascades_sub_list

    # Handle the last time step (which includes all interactions)
    t_cascades_sub_list = t_cascades_pd.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
    sub_timesas = t_cascades_pd["timestamp"].max()
    t_cascades_list[sub_timesas] = t_cascades_sub_list

    # Initialize a dictionary to store the dynamic diffusion graph at each time step
    dynamic_graph_dict_list = dict()

    # Iterate through sorted timestamps and construct the diffusion graph
    for key in sorted(t_cascades_list.keys()):
        edges_list = t_cascades_list[key]

        # Create a dictionary to track interactions between user pairs
        cascade_dic = defaultdict(list)
        for upair in edges_list:
            cascade_dic[upair].append(1)

        # Store the diffusion graph at the current time step
        dynamic_graph_dict_list[key] = cascade_dic

    return dynamic_graph_dict_list
