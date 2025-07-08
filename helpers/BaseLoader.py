import logging
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from utils import Constants  # Assuming Constants.EOS and Constants.PAD are defined


class CascadeDataset(Dataset):
    """
    A PyTorch-compatible Dataset for cascade sequences.
    """

    def __init__(self, cascades, timestamps, indices):
        self.cascades = cascades
        self.timestamps = timestamps
        self.indices = indices

    def __len__(self):
        return len(self.cascades)

    def __getitem__(self, idx):
        # Returns a single, unpadded data point.
        return (
            self.cascades[idx],
            self.timestamps[idx],
            self.indices[idx]
        )


def collate_fn(batch):
    """
    Collates a batch of samples into padded Tensors.
    This function replaces the padding logic from the old custom DataLoader.

    Args:
        batch (list): A list of tuples, where each tuple is a return value from
                      CascadeDataset.__getitem__, i.e., (cascade, timestamp, index).

    Returns:
        A tuple of padded Tensors: (seq_data, seq_data_timestamp, seq_idx).
    """
    cascades, timestamps, indices = zip(*batch)

    max_len = 200  # Fixed maximum sequence length

    # --- Pad user sequences ---
    pad_value_user = Constants.PAD
    padded_cascades = []
    for seq in cascades:
        if len(seq) < max_len:
            padded_seq = seq + [pad_value_user] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_cascades.append(padded_seq)

    seq_data = torch.LongTensor(padded_cascades)

    # --- Pad timestamp sequences ---
    pad_value_time = float(Constants.PAD)
    padded_timestamps = []
    for seq in timestamps:
        if len(seq) < max_len:
            padded_seq = seq + [pad_value_time] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_timestamps.append(padded_seq)

    seq_data_timestamp = torch.FloatTensor(padded_timestamps)

    # --- Convert indices ---
    seq_idx = torch.LongTensor(indices)

    return seq_data, seq_data_timestamp, seq_idx


class BaseLoader(object):
    """
    Handles loading, processing, and splitting the cascade dataset.
    """

    def __init__(self, args):
        self.test_set = None
        self.valid_set = None
        self.train_set = None
        # Set file paths based on the dataset name
        self.data_name = args.data_name
        self.data = f'data/{self.data_name}/cascades.txt'
        self.u2idx_dict = f'data/{self.data_name}/u2idx.pickle'
        self.idx2u_dict = f'data/{self.data_name}/idx2u.pickle'
        self.net_data = f'data/{self.data_name}/edges.txt'

        self.user_num = 0
        self.cas_num = 0
        self.min_cascade_len = args.filter_num

    def split_data(self, train_ratio=0.8, valid_ratio=0.1, load_dict=True, with_eos=True):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            train_ratio (float): Proportion of data to be used for training.
            valid_ratio (float): Proportion of data to be used for validation.
            load_dict (bool): Whether to load user-index mappings from files.
            with_eos (bool): Whether to append an end-of-sequence (EOS) token to each cascade.

        Returns:
            tuple: A tuple containing user count, original cascades, original timestamps,
                   and the split datasets (train_set, valid_set, test_set) as CascadeDataset objects.
        """
        if not load_dict:
            # Build user-to-index mappings if not loading from files
            user_count, user_to_index, index_to_user = self.build_index(self.data)
            with open(self.u2idx_dict, 'wb') as f:
                pickle.dump(user_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.idx2u_dict, 'wb') as f:
                pickle.dump(index_to_user, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Load mappings from existing files
            with open(self.u2idx_dict, 'rb') as f:
                user_to_index = pickle.load(f)
            with open(self.idx2u_dict, 'rb') as f:
                index_to_user = pickle.load(f)
            user_count = len(user_to_index)

        cascades, timestamps = [], []

        # Read and parse the cascade data file
        for line in open(self.data):
            if not line.strip():
                continue

            current_timestamps = []
            current_users = []
            chunks = line.strip().split(',')

            for chunk in chunks:
                processed_chunk = chunk.strip()
                if not processed_chunk:
                    continue
                try:
                    parts = processed_chunk.split()
                    user, timestamp = None, None
                    # Handle different data formats
                    if len(parts) == 2:  # Format: user timestamp (e.g., Twitter, Douban)
                        user, timestamp = parts
                    elif len(parts) == 3:  # Format: root_user user timestamp (e.g., Android, Christianity)
                        root, user, timestamp = parts
                        if root in user_to_index:
                            current_users.append(user_to_index[root])
                            current_timestamps.append(float(timestamp))
                    else:
                        logging.warning(f"Unexpected chunk format: {chunk} in line: {line.strip()}")
                        continue

                except Exception as e:
                    logging.error(f"Error processing chunk: {chunk} -> {e}")
                    continue

                if user and user in user_to_index:
                    current_users.append(user_to_index[user])
                    current_timestamps.append(float(timestamp))

            # Filter cascades by length
            if self.min_cascade_len <= len(current_users) <= 500:
                if with_eos:
                    current_users.append(Constants.EOS)
                    current_timestamps.append(Constants.EOS)  # Assuming EOS for timestamp is defined
                cascades.append(current_users)
                timestamps.append(current_timestamps)

        # Sort cascades chronologically based on the first timestamp
        sorted_indices = sorted(
            range(len(timestamps)),
            key=lambda i: timestamps[i][0] if timestamps[i] and isinstance(timestamps[i][0], (int, float)) else float(
                'inf')
        )
        cascades = [cascades[i] for i in sorted_indices]
        timestamps = [timestamps[i] for i in sorted_indices]

        # Split data into training, validation, and test sets
        total_cascades = len(cascades)
        train_end = int(train_ratio * total_cascades)
        valid_end = int((train_ratio + valid_ratio) * total_cascades)

        # Return CascadeDataset objects instead of raw tuples
        train_set = CascadeDataset(
            cascades[:train_end],
            timestamps[:train_end],
            sorted_indices[:train_end]
        )
        valid_set = CascadeDataset(
            cascades[train_end:valid_end],
            timestamps[train_end:valid_end],
            sorted_indices[train_end:valid_end]
        )
        test_set = CascadeDataset(
            cascades[valid_end:],
            timestamps[valid_end:],
            sorted_indices[valid_end:]
        )

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        self.user_num = user_count
        self.cas_num = total_cascades

        # Log dataset statistics using the length of the new Dataset objects
        logging.info(
            f"\nTraining size: {len(train_set)}\nValidation size: {len(valid_set)}\nTesting size: {len(test_set)}")

        self.cascades = cascades
        self.timestamps = timestamps

        self.all_cascades = cascades[:valid_end]  # Designed for MIMDP's Hierarchical Knowledge Distillation

        # Note: self.train_cas_user_dict needs the original tuple format for setup
        train_cascades_tuple = (cascades[:train_end], timestamps[:train_end], sorted_indices[:train_end])
        self.train_cas_user_dict = self.create_cascade_user_dict(train_cascades_tuple)

        return user_count, cascades, timestamps, train_set, valid_set, test_set

    def log_dataset_stats(self, cascades, with_eos, train_set, valid_set, test_set):
        """Logs detailed statistics of the processed dataset."""
        # Note: This function expects dataset splits as tuples of lists, e.g., train_set[0]
        logging.info(
            f"\nTraining size: {len(train_set[0])}\nValidation size: {len(valid_set[0])}\nTesting size: {len(test_set[0])}")

        num_actual_users = self.user_num - 2  # Exclude <blank> and </s>
        num_filtered_cascades = self.cas_num

        total_interactions_actual = 0
        actual_lengths = []

        if num_filtered_cascades > 0:
            for cas_sequence in cascades:
                # Calculate length excluding EOS token
                current_actual_len = len(cas_sequence)
                if with_eos and current_actual_len > 0 and cas_sequence[-1] == Constants.EOS:
                    current_actual_len -= 1
                total_interactions_actual += current_actual_len
                if current_actual_len > 0:
                    actual_lengths.append(current_actual_len)

        avg_length_actual = (total_interactions_actual / num_filtered_cascades) if num_filtered_cascades > 0 else 0.0
        density = (total_interactions_actual / (
                num_actual_users * num_filtered_cascades)) if num_filtered_cascades > 0 and num_actual_users > 0 else 0.0

        logging.info("--- Dataset Statistics Summary ---")
        logging.info(f"Total Unique Users (excluding special tokens): {num_actual_users}")
        logging.info(f"Total Cascades (after filtering): {num_filtered_cascades}")
        logging.info(f"Total Interactions (user participations): {total_interactions_actual}")
        logging.info(f"Avg. Cascade Length (users per cascade): {avg_length_actual:.2f}")
        logging.info(f"Max. Cascade Length (users): {max(actual_lengths) if actual_lengths else 0}")
        logging.info(f"Min. Cascade Length (users, >=filter_num): {min(actual_lengths) if actual_lengths else 0}")
        logging.info(f"Density (Interactions / (Users * Cascades)): {density:.6f}")
        logging.info("----------------------------------")

    def get_dataloaders(self, args):
        """
        Creates and returns the data loaders for training, validation, and testing.
        This is the standard implementation using TorchDataLoader.
        """
        logging.info("Preparing standard TorchDataLoaders...")

        # 调用 split_data 获取数据集
        user_size, _, _, train_dataset, valid_dataset, test_dataset = self.split_data(
            args.train_rate,
            args.valid_rate,
            load_dict=True
        )
        # 将 user_num 存起来，模型初始化时可能需要
        self.user_num = user_size

        use_cuda = args.gpu and torch.cuda.is_available()

        train_data = TorchDataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=use_cuda, collate_fn=collate_fn
        )
        valid_data = TorchDataLoader(
            dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=use_cuda, collate_fn=collate_fn
        )
        test_data = TorchDataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=use_cuda, collate_fn=collate_fn
        )

        # 返回准备好的 loaders
        return train_data, valid_data, test_data

    def create_cascade_user_dict(self, train_set):
        """
        Creates a dictionary mapping cascade indices to their user lists.

        Args:
            train_set (tuple): Training data from split_data, containing cascades, timestamps, and indices.

        Returns:
            dict: A dictionary with cascade indices as keys and lists of users as values.
        """
        cascades, _, indices = train_set
        cascade_dict = {}
        for i, cascade in zip(indices, cascades):
            # Create a user list for the cascade, excluding the EOS token
            user_list = [user for user in cascade if user != Constants.EOS]
            cascade_dict[i] = user_list
        return cascade_dict

    def build_index(self, data_path):
        """
        Builds user-to-index and index-to-user mappings from the data file.
        """
        user_set = set()

        for line_id, line in enumerate(open(data_path), 1):
            if not line.strip():
                continue
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    user = None
                    parts = chunk.split()
                    # Process different formats: (user, timestamp) or (root, user, timestamp)
                    if len(parts) == 2:
                        user, _ = parts
                    elif len(parts) == 3:
                        root, user, _ = parts
                        user_set.add(root)
                except Exception:
                    logging.error(f"Error parsing chunk: '{chunk}' on line {line_id} during build_index.")
                    logging.error(f"Original line content: {line.strip()}")
                    continue
                if user:
                    user_set.add(user)

        # Initialize mappings with special tokens
        u2idx = {'<blank>': 0, '</s>': 1}
        idx2u = ['<blank>', '</s>']

        # Assign indices to sorted users for deterministic mapping
        for user_val in sorted(list(user_set)):
            pos = len(u2idx)
            u2idx[user_val] = pos
            idx2u.append(user_val)

        user_size = len(idx2u)
        logging.info(f"user_size (total unique users in data): {user_size - 2}")
        return user_size, u2idx, idx2u
