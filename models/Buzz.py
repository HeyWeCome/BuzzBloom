# -*- coding: utf-8 -*
import math
import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.data import Data
from layers import GraphBuilder
from utils import Constants
from torch.autograd import Variable
from layers.Commons import DynamicGraphNN, GraphNN, Fusion, TimeAttention
from layers.TransformerBlock import TransformerBlock

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Buzz(nn.Module):
    """
    The Buzz model.

    An Information Diffusion Prediction Model Aligning Multiple Propagation Intentions with Dynamic User Cognition.
    IEEE Transactions on Computational Social Systems, 2025.
    """

    @staticmethod
    def parse_model_args(parser):
        """
        Adds model-specific arguments to the parser.

        Args:
            parser: an argparse.ArgumentParser object.

        Returns:
            The parser with added arguments.
        """
        parser.add_argument('--pos_dim', type=int, default=64,
                            help='Dimension for positional encoding. The original paper used 8.')
        parser.add_argument('--n_heads', type=int, default=8,
                            help='Number of attention heads in the Transformer block. The original paper used 8.')
        parser.add_argument('--time_step_split', type=int, default=8,
                            help='Number of time windows to split the dynamic graph into.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of latent intentions to disentangle.')
        return parser

    def __init__(self, args, data_loader):
        """
        Initializes the Buzz model.

        Args:
            args: Command-line arguments containing model hyperparameters.
            data_loader: The data loader object, providing access to dataset statistics
                         like user_num, cas_num, and file paths.
        """
        super(Buzz, self).__init__()
        self.user_num = data_loader.user_num
        self.embedding_size = args.d_model
        self.pos_dim = args.pos_dim
        self.n_heads = args.n_heads
        self.time_step_split = args.time_step_split
        self.device = args.device
        self.dropout = nn.Dropout(args.dropout)
        self.drop_timestamp = nn.Dropout(args.dropout) # Dropout for timestamps, though not used in the final forward pass.

        # --- Embedding Layers ---
        # In the original paper, a dropout of 0.5 is often used for GNNs.
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim) # Positional embeddings for sequence order.

        # --- Graph Neural Network Layers ---
        self.gnn_layer = GraphNN(self.user_num, self.embedding_size, dropout=0.5) # A standard GNN layer (potentially for a static graph).
        self.gnn_diffusion_layer = DynamicGraphNN(self.user_num, self.embedding_size, self.time_step_split) # GNN for processing dynamic graphs.

        # --- Attention and Output Layers ---
        self.decoder_attention = TransformerBlock(input_size=self.embedding_size, n_heads=self.n_heads)
        self.linear = nn.Linear(self.embedding_size, self.user_num) # Final projection layer to user vocabulary.

        # --- Graph Construction ---
        # Pre-builds the dynamic graph snapshots from the dataset.
        self.diffusion_graph = self._build_dynamic_heterogeneous_graph(data_loader, self.time_step_split)
        # self.diffusion_graph = GraphBuilder.build_dynamic_heterogeneous_graph(data_loader, self.time_step_split)

        # --- Multi-Intention Module ---
        # Components to extract K latent user intentions from the cascade history.
        self.attn_size = 8 # Intermediate dimension for the intention attention mechanism.
        self.K = args.K # Number of intentions.
        self.W1 = nn.Linear(self.embedding_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)
        self.linear2 = nn.Linear(self.embedding_size, self.embedding_size) # Another linear layer, not used in the final forward pass.

        # --- Weight Initialization ---
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights for specific layers of the model.
        """
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def _build_dynamic_heterogeneous_graph(self, dataloader, time_step_split):
        """
        Builds a series of dynamic heterogeneous graph snapshots.
        Each snapshot combines a static social network with a cumulative diffusion network
        up to a certain point in time.

        Args:
            dataloader: The data loader object.
            time_step_split: The number of time windows.

        Returns:
            A dictionary where keys are timestamps and values are torch_geometric.Data objects
            representing the graph at that time.
        """
        # Initialize dictionaries to store user-to-index and index-to-user mappings.
        _u2idx = {}  # Maps user ID to a unique integer index.
        _idx2u = []  # Maps integer index back to user ID.

        # Load the user-to-index mapping from the pre-processed file.
        with open(dataloader.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)

        # Load the index-to-user mapping.
        with open(dataloader.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        # --- Load Static Social Network (Follow relations) ---
        follow_relation = []  # List to store follow relationships (as directed edges).
        if os.path.exists(dataloader.net_data):
            with open(dataloader.net_data, 'r') as handle:
                # Read all edges from the social network file.
                edges_list = handle.read().strip().split("\n")
                edges_list = [edge.split(',') for edge in edges_list]
                # Convert user IDs to indices and store the follow relations.
                follow_relation = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in edges_list if
                                   edge[0] in _u2idx and edge[1] in _u2idx]

        # --- Load Dynamic Diffusion Graphs ---
        # This returns a dictionary mapping timestamps to cumulative lists of diffusion edges.
        dy_diff_graph_list = self._load_dynamic_diffusion_graph(dataloader, time_step_split)

        dynamic_graph = dict()  # Initialize the dictionary to store final graph snapshots.

        # Iterate through each time-stamped diffusion graph.
        for x in sorted(dy_diff_graph_list.keys()):
            # Start with the static follow relations for the current snapshot.
            edges_list = follow_relation
            # Assign edge type 0 to all follow relations.
            edges_type_list = [0] * len(follow_relation)
            # Assign a default weight of 1.0 to follow relations.
            edges_weight = [1.0] * len(follow_relation)

            # Add the dynamic diffusion relations (e.g., retweets) to the current snapshot.
            for key, value in dy_diff_graph_list[x].items():
                edges_list.append(key)  # Add the (user1, user2) diffusion edge.
                edges_type_list.append(1)  # Assign edge type 1 to diffusion relations.
                # The weight can represent the frequency of this interaction.
                edges_weight.append(sum(value))

            # Convert lists to PyTorch tensors.
            edges_list_tensor = torch.LongTensor(edges_list).t() # Shape: [2, num_edges]
            edges_type = torch.LongTensor(edges_type_list)
            edges_weight = torch.FloatTensor(edges_weight)

            # Create a PyTorch Geometric Data object for the snapshot.
            data = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_weight=edges_weight)
            # Store the graph snapshot in the dictionary, keyed by its timestamp.
            dynamic_graph[x] = data

        return dynamic_graph

    def _load_dynamic_diffusion_graph(self, dataloader, time_step_split):
        """
        Loads and processes raw cascade data to create cumulative diffusion graphs for different time steps.

        Args:
            dataloader: The data loader object.
            time_step_split: The number of time windows.

        Returns:
            A dictionary where keys are timestamps and values are dictionaries of diffusion edges.
        """
        # Load user-to-index and index-to-user mappings.
        _u2idx = {}
        _idx2u = []
        with open(dataloader.u2idx_dict, 'rb') as handle:
            _u2idx = pickle.load(handle)
        with open(dataloader.idx2u_dict, 'rb') as handle:
            _idx2u = pickle.load(handle)

        # Extract cascades, timestamps, and labels from the training data.
        dataset = dataloader.train_set
        cascades = dataset.cascades
        timestamps = dataset.timestamps

        t_cascades = [] # List to store all user-user interaction pairs with their timestamps.

        # Process each cascade to extract sequential user pairs.
        for cascade, timestamp in zip(cascades, timestamps):
            userlist = list(zip(cascade, timestamp))

            # Example:
            # If userlist is [(user_A, time_1), (user_B, time_2), (user_C, time_3)],
            # zip(userlist[:-1], userlist[1:]) creates pairs:
            # [((user_A, time_1), (user_B, time_2)), ((user_B, time_2), (user_C, time_3))]
            # pair_user will then be [(user_A, user_B, time_2), (user_B, user_C, time_3)],
            # capturing the directed information flow.
            pair_user = [(i[0], j[0], j[1]) for i, j in zip(userlist[:-1], userlist[1:])]

            # Filter cascades by length to avoid very short or extremely long ones.
            if len(pair_user) > 1 and len(pair_user) <= 500:
                t_cascades.extend(pair_user)

        # Create a pandas DataFrame for easier manipulation.
        t_cascades_pd = pd.DataFrame(t_cascades, columns=["user1", "user2", "timestamp"])

        # Sort all interactions globally by their timestamp.
        t_cascades_pd = t_cascades_pd.sort_values(by="timestamp")

        # Calculate the number of interactions per time window.
        t_cascades_length = t_cascades_pd.shape[0]
        step_length_x = t_cascades_length // time_step_split

        t_cascades_list = dict() # Dictionary to store cumulative edges for each time step.
        accumulated_edges = [] # List to accumulate edges over time.

        # Create cumulative snapshots at each time step boundary.
        for x in range(step_length_x, t_cascades_length - step_length_x, step_length_x):
            # Get all data up to the current time boundary 'x'.
            t_cascades_pd_sub = t_cascades_pd[:x]
            # Extract user pairs from this subset.
            t_cascades_sub_list = t_cascades_pd_sub.apply(lambda row: (row["user1"], row["user2"]), axis=1).tolist()
            # Accumulate the edges.
            accumulated_edges.extend(t_cascades_sub_list)
            # Use the max timestamp of the subset as the key for this snapshot.
            sub_timestamp = t_cascades_pd_sub["timestamp"].max()
            # Store the unique set of accumulated edges.
            t_cascades_list[sub_timestamp] = list(set(accumulated_edges))

        # Handle the final time step, which includes all interactions.
        t_cascades_sub_list = t_cascades_pd.apply(lambda row: (row["user1"], row["user2"]), axis=1).tolist()
        accumulated_edges.extend(t_cascades_sub_list)
        sub_timestamp = t_cascades_pd["timestamp"].max()
        t_cascades_list[sub_timestamp] = list(set(accumulated_edges))

        dynamic_graph_dict_list = dict() # Final dictionary to return.

        # Convert the edge lists into a format suitable for the next building step.
        for key in sorted(t_cascades_list.keys()):
            edges_list = t_cascades_list[key]
            # Use a defaultdict to count occurrences of each edge, which can be used as weight.
            cascade_dic = defaultdict(list)
            for upair in edges_list:
                cascade_dic[upair].append(1)
            dynamic_graph_dict_list[key] = cascade_dic

        return dynamic_graph_dict_list

    def forward(self, input_seq, input_timestamp, tgt_idx):
        """
        Defines the forward pass of the Buzz model.

        Args:
            input_seq (Tensor): Batch of input user sequences. Shape: [batch_size, seq_len].
            input_timestamp (Tensor): Batch of corresponding timestamps. Shape: [batch_size, seq_len].
            tgt_idx (Tensor): Not used in this implementation but might be for other tasks.

        Returns:
            Tensor: Logits for the next user prediction. Shape: [batch_size * (seq_len-1), user_num].
        """
        # --- 1. Input Preparation ---
        # For next-item prediction, use all but the last element as input.
        input_seq = input_seq[:, :-1]  # Shape: [batch_size, seq_len-1]
        input_timestamp = input_timestamp[:, :-1]  # Shape: [batch_size, seq_len-1]

        # Create a padding mask to ignore PAD tokens in attention calculations.
        mask = (input_seq == Constants.PAD)  # Shape: [batch_size, seq_len-1]

        # Generate positional embeddings for each position in the sequence.
        batch_t = torch.arange(input_seq.size(1), device=self.device).expand(input_seq.size())
        position_embed = self.dropout(self.pos_embedding(batch_t))  # Shape: [batch_size, seq_len-1, pos_dim]

        batch_size, max_len = input_seq.size()

        # --- 2. Dynamic Graph-based Embeddings ---
        # First, compute node embeddings for each discrete graph snapshot using the GNN layer.
        # This results in a dictionary mapping each timestamp to a user embedding matrix.
        dynamic_node_emb_dict = self.gnn_diffusion_layer(self.diffusion_graph) # {timestamp: [user_num, hidden_size]}

        # For each graph snapshot, get the embeddings for all users in the input sequences.
        dyuser_emb_list = []
        for val in sorted(dynamic_node_emb_dict.keys()):
            # Look up embeddings for the input sequence from the current snapshot's embedding matrix.
            dyuser_emb_sub = F.embedding(input_seq, dynamic_node_emb_dict[val].to(self.device)).unsqueeze(2)
            dyuser_emb_list.append(dyuser_emb_sub)

        # Concatenate embeddings from all snapshots.
        dyuser_emb = torch.cat(dyuser_emb_list, dim=2)  # Shape: [bth, seq_len, time_step_split, hidden_size]

        # Average the embeddings across the time dimension to get a single dynamic representation.
        dyemb = dyuser_emb.mean(dim=2)  # Shape: [bth, seq_len, hidden_size]
        dyemb = self.dropout(dyemb)

        # --- 3. Static and Intention-based Embeddings ---
        # Get the standard static user embeddings and add positional information.
        original_seq_emb = self.user_embedding(input_seq)
        original_seq_emb += position_embed

        # --- Intention Extraction ---
        valid_his = (input_seq > 0).long() # Mask for non-padding tokens.
        # Calculate attention scores for K intentions over the sequence history.
        attn_score = self.W2(self.W1(original_seq_emb).tanh())  # [bsz, seq_len, K]
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf) # Mask padding.
        attn_score = attn_score.transpose(-1, -2) # [bsz, K, seq_len]
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0) # Handle potential NaNs.
        # Compute the K intention vectors by weighted sum of static embeddings.
        intention_vectors = (original_seq_emb.unsqueeze(1) * attn_score.unsqueeze(-1)).sum(-2) # [bsz, K, emb]
        # Take the sum over the K intentions to get a single intention vector per sequence.
        intention_vectors = intention_vectors.sum(1) # [bsz, emb]

        # Expand intention vector to match the sequence length for the attention mechanism.
        intention_vectors = intention_vectors.unsqueeze(1).expand(-1, max_len, -1) # [bsz, seq_len, emb]

        # --- Fusion via Cross-Attention ---
        # Query: Dynamic embeddings; Key/Value: Intention vectors.
        # <--- 已修改：使用正确的参数名 Q, K, V
        att_out = self.decoder_attention(Q=dyemb,
                                         K=intention_vectors,
                                         V=intention_vectors,
                                         mask=mask)  # [batch_size, seq_len, hidden_size]
        att_out = self.dropout(att_out)

        # --- Prediction ---
        output = self.linear(att_out)  # [batch_size, seq_len, user_num]
        # Apply a mask to prevent predicting users already in the sequence history.
        mask = self.get_previous_user_mask(input_seq, self.user_num)
        output = output + mask
        return output.view(-1, output.size(-1))

    def get_previous_user_mask(self, seq, user_size):
        """
        Creates a mask to prevent the model from predicting users that have already
        appeared in the input sequence up to the current time step.

        Args:
            seq (Tensor): The input user sequence. Shape: [batch_size, seq_len].
            user_size (int): The total number of users (vocabulary size).

        Returns:
            Tensor: A mask with large negative values at the positions of previous users.
                    Shape: [batch_size, seq_len, user_size].
        """
        assert seq.dim() == 2
        device = seq.device
        # Create a lower triangular matrix to capture history.
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        # Repeat the sequence to create a [batch, seq_len, seq_len] tensor.
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = torch.from_numpy(np.tril(np.ones(prev_shape), k=-1)).float().to(device)

        # By multiplying with the lower triangular mask, for each time step `t`,
        # we get a list of all users that appeared before `t`.
        masked_seq = previous_mask * seqs.data.float()

        # The rest of the logic uses `scatter_` to create the final mask.
        # It creates a zero tensor of shape [batch, seq_len, user_size] and places
        # a large negative value at the indices corresponding to users in `masked_seq`.
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size, device=device)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1e9)) # Use a large negative number
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq


    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        """
        Calculates the loss and the number of correct predictions for a batch.

        Args:
            input_seq (Tensor): Batch of input sequences.
            input_seq_timestamp (Tensor): Batch of input timestamps.
            history_seq_idx (Tensor): Not used here.
            loss_func: The loss function (e.g., CrossEntropyLoss).
            gold (Tensor): The ground truth target sequences.

        Returns:
            tuple: A tuple containing the loss (float) and the number of correct predictions (float).
        """
        # Get the model's predictions.
        pred = self.forward(input_seq, input_seq_timestamp, history_seq_idx)

        # Reshape the ground truth tensor to match the prediction tensor's shape.
        # gold.contiguous().view(-1) : [batch_size, seq_len-1] -> [batch_size * (seq_len-1)]
        gold_flat = gold.contiguous().view(-1)

        # Calculate the loss between predictions and ground truth.
        loss = loss_func(pred, gold_flat)

        # --- Calculate Accuracy ---
        # Get the index of the highest score for each prediction.
        pred_flat = pred.max(1)[1]
        # Compare predictions with the ground truth.
        n_correct = pred_flat.data.eq(gold_flat.data)
        # Mask out the padding tokens so they don't contribute to the accuracy calculation.
        non_pad_mask = gold_flat.ne(Constants.PAD).data
        n_correct = n_correct.masked_select(non_pad_mask).sum().float()

        return loss, n_correct
