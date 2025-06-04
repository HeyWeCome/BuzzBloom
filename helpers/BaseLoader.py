# BaseLoader.py

import logging
import random
import numpy as np
import torch
# from torch.autograd import Variable # Variable is deprecated, use torch.Tensor
from utils import Constants
import pickle


class BaseLoader(object):
    def __init__(self, args):
        # 根据数据集名称设置各类文件路径
        self.data_name = args.data_name
        self.data = 'data/' + self.data_name + '/cascades.txt'  # 数据文件路径
        self.u2idx_dict = 'data/' + self.data_name + '/u2idx.pickle'  # 用户到索引的映射字典
        self.idx2u_dict = 'data/' + self.data_name + '/idx2u.pickle'  # 索引到用户的映射字典
        self.save_path = ''  # 保存路径，初始为空
        self.net_data = 'data/' + self.data_name + '/edges.txt'  # 网络数据路径
        self.user_num = 0
        self.cas_num = 0
        self.device = args.device  # Store device from args

    def split_data(self, train_ratio=0.8, valid_ratio=0.1, load_dict=True, with_eos=True):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            train_rate (float): Proportion of data to be used for training.
            valid_rate (float): Proportion of data to be used for validation.
            load_dict (bool): Whether to load user index mapping from file.
            with_eos (bool): Whether to append end-of-sequence token.

        Returns:
            tuple: User size, cascades data, timestamps, training data, validation data, test data.
        """
        # split_data function to split the dataset into training, validation, and test sets
        user_to_index = {}  # Mapping from user to index
        index_to_user = []  # List of users by index

        if not load_dict:
            # Build the index if not loading from a file
            user_count, user_to_index, index_to_user = self.build_index(self.data)
            # Save the mapping dictionaries to files
            with open(self.u2idx_dict, 'wb') as f:
                pickle.dump(user_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.idx2u_dict, 'wb') as f:
                pickle.dump(index_to_user, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Load the mapping from files
            with open(self.u2idx_dict, 'rb') as f:
                user_to_index = pickle.load(f)
            with open(self.idx2u_dict, 'rb') as f:
                index_to_user = pickle.load(f)
            user_count = len(user_to_index)  # Count of unique users

        cascades, timestamps = [], []  # Store time-sequenced cascade data and timestamps

        # Read the data file line by line
        for line in open(self.data):
            if not line.strip():
                continue  # Skip empty lines

            current_timestamps = []  # Timestamps for the current line
            current_users = []  # Users for the current line
            chunks = line.strip().split(',')  # Split line data by comma

            for chunk in chunks:
                processed_chunk = chunk.strip()  # 先strip掉chunk两端的空格
                if not processed_chunk:  # 如果chunk处理后是空字符串，则跳过
                    continue
                try:
                    # Handle different formats of user and timestamp data
                    parts = processed_chunk.split() # 使用处理过的chunk
                    # Twitter, Douban
                    if len(parts) == 2:  # Format: user timestamp
                        user, timestamp = parts
                    # Android, Christianity
                    elif len(parts) == 3:  # Format: root_user user timestamp
                        root, user, timestamp = parts
                        if root in user_to_index:
                            current_users.append(user_to_index[root])  # Append root user index
                            current_timestamps.append(float(timestamp))  # Append timestamp
                    else:  # Added to handle cases where parts might be < 2
                        logging.warning(f"Skipping malformed chunk: {chunk} in line: {line.strip()}")
                        continue


                except Exception as e:
                    print(f"Error processing chunk: {chunk} -> {e}")  # Print any errors
                    continue  # Skip to next chunk on error

                if user in user_to_index:
                    current_users.append(user_to_index[user])  # Append user index
                    current_timestamps.append(float(timestamp))  # Append timestamp

            # Filter cascades based on user count
            if 1 <= len(current_users) <= 500:
                if with_eos:
                    current_users.append(Constants.EOS)  # Append end-of-sequence marker
                    current_timestamps.append(Constants.EOS_TIME if hasattr(Constants,
                                                                            'EOS_TIME') else -1.0)  # Use a defined EOS time or a placeholder
                cascades.append(current_users)  # Add current cascade
                timestamps.append(current_timestamps)  # Add current timestamps

        # Sort cascades by timestamp (ensure timestamps[i] is not empty and contains valid numbers)
        # Filter out empty timestamp lists before sorting to avoid errors
        valid_indices_for_sorting = [i for i, ts_list in enumerate(timestamps) if
                                     ts_list and isinstance(ts_list[0], (int, float))]

        # Apply filtering
        cascades = [cascades[i] for i in valid_indices_for_sorting]
        timestamps = [timestamps[i] for i in valid_indices_for_sorting]

        # Now sort based on the first valid timestamp
        sorted_indices_map = sorted(range(len(timestamps)),
                                    key=lambda i: timestamps[i][0] if timestamps[i] else float('inf'))

        cascades = [cascades[i] for i in sorted_indices_map]
        timestamps = [timestamps[i] for i in sorted_indices_map]
        # The original `sorted_indices` seemed to map back to the original, unsorted, full list of cascades.
        # If you need to preserve original indices for some reason, this part needs careful re-evaluation.
        # For now, assuming sorted_indices should correspond to the *final* sorted `cascades` list:
        original_indices_after_filtering_and_sorting = [valid_indices_for_sorting[i] for i in sorted_indices_map]

        # Split data into train, validation, and test sets
        total_cascades = len(cascades)
        train_end = int(train_ratio * total_cascades)  # End index for training set
        valid_end = int((train_ratio + valid_ratio) * total_cascades)  # End index for validation set

        # We need to use the original_indices_after_filtering_and_sorting if `idx` in DataLoader refers to original cascade IDs
        train_set = (
            cascades[:train_end],
            timestamps[:train_end],
            original_indices_after_filtering_and_sorting[:train_end]
        )
        valid_set = (
            cascades[train_end:valid_end],
            timestamps[train_end:valid_end],
            original_indices_after_filtering_and_sorting[train_end:valid_end]
        )
        test_set = (
            cascades[valid_end:],
            timestamps[valid_end:],
            original_indices_after_filtering_and_sorting[valid_end:]
        )

        # Shuffle training set data (shuffle all three lists consistently)
        # temp_train = list(zip(train_set[0], train_set[1], train_set[2]))
        # random.shuffle(temp_train)
        # train_cas, train_time, train_idx = zip(*temp_train) if temp_train else ([], [], [])
        # train_set = (list(train_cas), list(train_time), list(train_idx))

        # It's usually better to shuffle the indices and then reorder, or shuffle within DataLoader
        # For simplicity, if `DataLoader` shuffles, this might not be strictly necessary here.
        # However, your original code shuffles each component independently, which is wrong.
        # Let's shuffle them together:

        combined_train = list(zip(train_set[0], train_set[1], train_set[2]))
        random.shuffle(combined_train)
        shuffled_train_cascades, shuffled_train_timestamps, shuffled_train_indices = zip(
            *combined_train) if combined_train else ([], [], [])
        train_set = (list(shuffled_train_cascades), list(shuffled_train_timestamps), list(shuffled_train_indices))

        # Print dataset statistics
        total_length = sum(len(cas) for cas in cascades if cas)  # Add check for non-empty cas
        logging.info(
            f"Training size: {len(train_set[1])}\nValidation size: {len(valid_set[1])}\nTesting size: {len(test_set[1])}")
        if total_cascades > 0:
            logging.info(f"Total size: {total_cascades}\nAverage length: {total_length / total_cascades:.2f}")
            logging.info(f"Maximum length: {max(len(cas) for cas in cascades if cas):.2f}")  # Add check
            logging.info(f"Minimum length: {min(len(cas) for cas in cascades if cas):.2f}")  # Add check
        else:
            logging.info("No cascades loaded or all cascades were filtered out.")

        self.user_num = user_count
        self.cas_num = len(cascades)
        logging.info(
            f"User size: {self.user_num - 2 if self.user_num >= 2 else 0}")  # Print user count excluding special markers
        logging.info(f"Cascade size: {self.cas_num}")

        self.cascades = cascades
        self.timestamps = timestamps
        self.train_data = train_set
        self.valid_data = valid_set
        self.test_data = test_set
        # Return user_count, and the *final processed* cascades and timestamps
        return user_count, self.cascades, self.timestamps, train_set, valid_set, test_set

    def build_index(self, data):
        # buildIndex函数用于构建用户到索引的映射和索引到用户的列表
        user_set = set()  # 用于存储唯一用户
        u2idx = {}  # 用户到索引的映射字典
        idx2u = []  # 索引到用户的列表

        line_id = 0  # 行编号
        for line in open(data):
            line_id += 1  # 增加行编号
            if len(line.strip()) == 0:
                continue  # 跳过空行
            chunks = line.strip().split(',')  # 按逗号分割行数据
            for chunk in chunks:
                try:
                    parts = chunk.split()  # Moved outside
                    # 处理不同格式的用户和时间戳
                    if len(parts) == 2:
                        user, timestamp = parts  # 格式：用户 时间戳
                    elif len(parts) == 3:
                        root, user, timestamp = parts  # 格式：根用户 用户 时间戳
                        user_set.add(root)  # 将根用户添加到集合中
                    else:  # Added to handle malformed chunks
                        logging.warning(f"Skipping malformed chunk in build_index: {chunk} in line_id: {line_id}")
                        continue
                except ValueError:  # More specific exception
                    # 捕获异常并输出出错的信息
                    logging.error(f"ValueError processing chunk: {chunk} in line: {line.strip()} at line_id: {line_id}")
                    continue  # Skip malformed chunk
                except Exception as e:
                    logging.error(
                        f"Generic error processing chunk: {chunk} in line: {line.strip()} at line_id: {line_id} - {e}")
                    continue
                user_set.add(user)  # 将用户添加到集合中

        pos = 0  # 初始化位置
        u2idx['<blank>'] = pos  # 添加空白标记
        idx2u.append('<blank>')  # 添加空白标记到列表
        pos += 1  # 更新位置
        u2idx['</s>'] = pos  # 添加结束标记
        idx2u.append('</s>')  # 添加结束标记到列表
        pos += 1  # 更新位置

        for user in user_set:
            u2idx[user] = pos  # 为每个用户分配索引
            idx2u.append(user)  # 将用户添加到索引列表
            pos += 1  # 更新位置

        user_size = len(user_set) + 2  # 计算用户总数（包括特殊标记）
        logging.info("user_size : %d" % (user_size - 2 if user_size >= 2 else 0))  # 打印用户数量
        return user_size, u2idx, idx2u  # 返回用户数量和映射字典、列表


class DataLoader(object):
    """ 用于数据迭代的类 """

    def __init__(
            self, cas, batch_size=64, load_dict=True, device=None, test=False, with_EOS=True):  # Changed cuda to device
        # 初始化DataLoader，设置批处理大小和其他参数
        self._batch_size = batch_size  # 批处理大小
        self.cas = cas[0]  # 级联数据
        self.time = cas[1]  # 时间戳
        self.idx = cas[2]  # 索引
        self.test = test  # 是否为测试模式
        self.with_EOS = with_EOS  # 是否使用结束标志

        # Store the device
        self.device = device if device is not None else torch.device('cpu')  # Default to CPU if not provided

        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))  # 计算批处理数量
        self._iter_count = 0  # 初始化迭代计数器

    def __iter__(self):
        return self  # 返回自身作为迭代器

    def __next__(self):
        return self.next()  # 获取下一个批处理

    def __len__(self):
        return self._n_batch  # 返回批处理的数量

    def next(self):
        """ 获取下一个批处理 """

        def pad_to_longest(insts, pad_value=Constants.PAD):  # Added pad_value for flexibility
            """ 将实例填充到批处理中最长序列长度 """
            # Determine max_len dynamically or use a fixed one if necessary
            # max_len = max(len(inst) for inst in insts) # Dynamic max_len for the batch
            max_len = 500  # Or keep fixed max_len as per original code

            # 对每个实例进行填充或截断
            # Ensure insts are lists of numbers. If they contain non-numeric EOS, handle it.
            processed_insts = []
            for inst in insts:
                # Convert non-numeric EOS to a numeric representation if needed, or ensure they are already numeric
                numeric_inst = [
                    item if isinstance(item, (int, float)) else (Constants.EOS if item == Constants.EOS else pad_value)
                    for item in inst]

                if len(numeric_inst) < max_len:
                    padded_inst = numeric_inst + [pad_value] * (max_len - len(numeric_inst))
                else:
                    padded_inst = numeric_inst[:max_len]
                processed_insts.append(padded_inst)

            inst_data = np.array(processed_insts)

            # 转换为PyTorch张量
            # `volatile` is deprecated. Use `torch.no_grad()` context for testing.
            # For training, gradients should be enabled by default.
            if self.test:
                with torch.no_grad():
                    # For timestamps, float might be more appropriate if they are not integers
                    if isinstance(pad_value, float) or any(
                            isinstance(x, float) for item_list in processed_insts for x in item_list):
                        inst_data_tensor = torch.FloatTensor(inst_data)
                    else:
                        inst_data_tensor = torch.LongTensor(inst_data)
            else:
                if isinstance(pad_value, float) or any(
                        isinstance(x, float) for item_list in processed_insts for x in item_list):
                    inst_data_tensor = torch.FloatTensor(inst_data)
                else:
                    inst_data_tensor = torch.LongTensor(inst_data)

            # Move tensor to the specified device
            inst_data_tensor = inst_data_tensor.to(self.device)

            return inst_data_tensor  # 返回填充后的张量

        if self._iter_count < self._n_batch:
            # 还有批次可迭代
            batch_idx = self._iter_count  # 获取当前批处理索引
            self._iter_count += 1  # 增加迭代计数器

            start_idx = batch_idx * self._batch_size  # 计算开始索引
            end_idx = (batch_idx + 1) * self._batch_size  # 计算结束索引

            seq_insts = self.cas[start_idx:end_idx]  # 获取序列实例
            seq_timestamp = self.time[start_idx:end_idx]  # 获取时间戳

            # Pad sequences (users/nodes are usually LongTensors)
            seq_data = pad_to_longest(seq_insts, pad_value=Constants.PAD)
            # Pad timestamps (timestamps can be FloatTensors)
            # Assuming Constants.EOS_TIME or a similar numeric placeholder for EOS in timestamps
            eos_time_value = Constants.EOS_TIME if hasattr(Constants, 'EOS_TIME') else -1.0  # Example
            seq_data_timestamp = pad_to_longest(seq_timestamp, pad_value=eos_time_value)

            # For indices, usually LongTensor and moved to device
            # `volatile` is deprecated.
            if self.test:
                with torch.no_grad():
                    seq_idx_tensor = torch.LongTensor(self.idx[start_idx:end_idx])
            else:
                seq_idx_tensor = torch.LongTensor(self.idx[start_idx:end_idx])

            seq_idx_tensor = seq_idx_tensor.to(self.device)

            return seq_data, seq_data_timestamp, seq_idx_tensor  # 返回序列数据、时间戳和索引
        else:
            # 没有更多批次可迭代，重置计数器
            self._iter_count = 0
            raise StopIteration()  # 引发StopIteration以结束迭代
