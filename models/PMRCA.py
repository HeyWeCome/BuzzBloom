#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：PMRCA.py
@Author     ：Heywecome
@Date       ：2025/6/30 09:15
@Description：PMRCA full model implementation.
"""

# Standard library imports
import math
import os

# Third-party imports
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torch_geometric.data import Data

# Local application imports
from layers import GraphBuilder
from layers.Commons import DynamicGraphNN, GraphNN, Fusion, TimeAttention
from layers.TransformerBlock import TransformerBlock
from utils import Constants

# This is a workaround for a known issue with some MKL/OpenMP library versions on certain OSes.
# It prevents a crash that can occur when multiple libraries are loaded.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PMRCA(nn.Module):
    # Implementation of the model described in:
    # He W, Xiao Y, Huang M, et al. A Pattern-Driven Information Diffusion Prediction Model Based on Multisource
    # Resonance and Cognitive Adaptation. Proceedings of the 48th International ACM SIGIR Conference on Research
    # and Development in Information Retrieval. 2025, doi: 10.1145/3726302.3729883
    """
    PMRCA (Pattern-driven Multisource Resonance and Cognitive Adaptation) model for information diffusion prediction.

    This model integrates graph neural networks with a transformer-based sequence model.
    It leverages multi-source resonance through a self-supervised contrastive loss between GCN layers
    and cognitive adaptation through a prototype-based contrastive loss to enhance user and cascade representations.
    
    The core of the PMRCA method is to optimize embedding quality through contrastive learning mechanisms.
    This approach is encoder-agnostic, allowing users to freely choose their preferred encoder.
    It is important to note that the performance of the selected encoder will directly impact the final
    performance of the model.
    """

    @staticmethod
    def parse_model_args(parser):
        """
        Adds model-specific arguments to the argument parser.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which to add arguments.

        Returns:
            argparse.ArgumentParser: The parser with added arguments.
        """
        parser.add_argument('--pos_dim', type=int, default=64,
                            help='Dimension for positional encoding.')
        parser.add_argument('--n_heads', type=int, default=8,
                            help='Number of attention heads in the transformer block.')
        parser.add_argument('--time_step_split', type=int, default=8,
                            help='Number of windows size (not directly used in this specific implementation).')
        parser.add_argument('--gcn_layers', type=int, default=3,
                            help='Number of GCN layers.')
        parser.add_argument('--ssl_reg', type=float, default=1e-7,
                            help='Regularization weight for the self-supervised inter-view loss.')
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='Weight for the item part of the SSL loss.')
        parser.add_argument('--ssl_temp', type=float, default=0.1,
                            help='Temperature parameter for the contrastive losses (SSL and ProtoNCE).')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of latent intentions for the multi-intention modeling.')
        parser.add_argument('--nc', type=int, default=10,  # Changed default from 1 to 10 to match original code's intent
                            help='Number of clusters/prototypes for cognitive adaptation.')
        return parser

    def __init__(self, args, data_loader):
        """
        Initializes the PMRCA model.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            data_loader (DataLoader): The data loader instance providing dataset details.
        """
        super(PMRCA, self).__init__()
        self.args = args
        self.device = args.device
        self.user_num = data_loader.user_num
        self.cas_num = data_loader.cas_num

        self.embedding_size = args.d_model
        self.pos_dim = args.pos_dim
        self.n_heads = args.n_heads

        # Embedding layers
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.cas_embedding = nn.Embedding(self.cas_num, self.embedding_size)
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)

        # Model components
        self.align_attention = TransformerBlock(input_size=self.embedding_size, n_heads=self.n_heads)
        self.seq_encoder = TransformerBlock(input_size=self.embedding_size, n_heads=self.n_heads)
        self.linear = nn.Linear(self.embedding_size, self.user_num)

        # Graph construction
        train_cas_user_dict = data_loader.train_cas_user_dict
        self.norm_adj = self.csr_to_tensor(self.build_adj_matrix(self.cas_num,
                                                                 self.user_num,
                                                                 train_cas_user_dict))
        self.gcn_layers = args.gcn_layers

        # Self-Supervised Learning (SSL) parameters
        self.ssl_reg = args.ssl_reg
        self.alpha = args.alpha
        self.ssl_temp = args.ssl_temp

        # Prototype-based contrastive learning parameters
        self.proto_reg = 8e-8  # The prototype-contrastive regularization weight.
        self.num_clusters = args.nc
        self.k = self.num_clusters  # Alias for number of clusters used in k-means
        self.user_centroids = None
        self.user_2cluster = None
        self.cas_centroids = None
        self.cas_2cluster = None

        self.init_weights()

    def init_weights(self):
        """Initializes weights for specific layers."""
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    def e_step(self):
        """
        Performs the E-step of the clustering algorithm.
        This method runs K-means on user and cascade embeddings to find centroids and assign nodes to clusters.
        It is called once per epoch before training starts.
        """
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        cas_embeddings = self.cas_embedding.weight.detach().cpu().numpy()

        # Check if CUDA is available for faiss-gpu.
        use_gpu = 'cuda' in str(self.device)
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, use_gpu)
        self.cas_centroids, self.cas_2cluster = self.run_kmeans(cas_embeddings, use_gpu)

    def run_kmeans(self, x, use_gpu=True):
        """
        Runs K-means clustering on the input data `x` using the faiss library.

        Args:
            x (np.ndarray): The data to cluster, with shape (n_samples, n_features).
            use_gpu (bool): Whether to use the GPU version of faiss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - centroids (torch.Tensor): The cluster centroids, normalized.
                - node2cluster (torch.Tensor): A tensor mapping each node to its cluster index.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss is not installed. Please install it for clustering.")

        # Set faiss to use a single thread to avoid OpenMP conflicts which can cause segmentation faults.
        faiss.omp_set_num_threads(1)

        kmeans = faiss.Kmeans(d=self.embedding_size, k=self.k, gpu=use_gpu, niter=20)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # Convert to specified device Tensors for subsequent computations
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    @staticmethod
    def build_adj_matrix(cas_count, user_count, train_dict, self_loop_flag=False):
        """
        Builds the adjacency matrix for a bipartite graph of cascades and users.

        Args:
            cas_count (int): The total number of cascades.
            user_count (int): The total number of users.
            train_dict (dict): A dictionary where keys are cascade IDs and values are lists of user IDs.
            self_loop_flag (bool): If True, adds self-loops to the adjacency matrix.

        Returns:
            sp.spmatrix: The adjacency matrix in sparse format.
        """
        # Create a bipartite user-cascade interaction matrix.
        R = sp.dok_matrix((cas_count, user_count), dtype=np.float32)
        for cas_id, user_list in train_dict.items():
            for user_id in user_list:
                R[cas_id, user_id] = 1
        R = R.tolil()

        # Create the full adjacency matrix for the graph (cascades + users).
        adj_mat = sp.dok_matrix((cas_count + user_count, cas_count + user_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # Populate the off-diagonal blocks with the interaction matrix.
        adj_mat[:cas_count, cas_count:] = R
        adj_mat[cas_count:, :cas_count] = R.T
        adj_mat = adj_mat.todok()

        if self_loop_flag:
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        return adj_mat

    def csr_to_tensor(self, matrix: sp.spmatrix) -> torch.sparse.FloatTensor:
        """
        Converts a scipy sparse matrix to a symmetrically normalized PyTorch sparse tensor.
        The normalization is D^-0.5 * A * D^-0.5.

        Args:
            matrix (sp.spmatrix): The sparse matrix to convert.

        Returns:
            torch.sparse.FloatTensor: The normalized sparse tensor.
        """
        rowsum = np.array(matrix.sum(1)) + 1e-10  # Add epsilon for numerical stability
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # Symmetrically normalize the matrix
        bi_lap = d_mat_inv_sqrt.dot(matrix).dot(d_mat_inv_sqrt)
        coo_bi_lap = bi_lap.tocoo()

        # Create the PyTorch sparse tensor
        i = torch.LongTensor(np.array([coo_bi_lap.row, coo_bi_lap.col]))
        data = torch.from_numpy(coo_bi_lap.data).float()
        sparse_bi_lap = torch.sparse_coo_tensor(i, data, torch.Size(coo_bi_lap.shape), dtype=torch.float32)

        return sparse_bi_lap.to(self.device)

    def get_ego_embeddings(self) -> torch.Tensor:
        """
        Retrieves the initial (0-hop) embeddings for all nodes (cascades and users).

        Returns:
            torch.Tensor: A tensor containing concatenated cascade and user embeddings.
        """
        user_embeddings = self.user_embedding.weight
        cas_embeddings = self.cas_embedding.weight
        ego_embeddings = torch.cat([cas_embeddings, user_embeddings], dim=0)
        return ego_embeddings

    def ssl_layer_loss(self, current_embedding, previous_embedding, cas_indices, user_indices):
        """
        Computes the self-supervised inter-view contrastive loss (InfoNCE).
        This loss encourages embeddings of the same node from different GCN layers (views)
        to be similar, while being dissimilar from other nodes' embeddings.

        Args:
            current_embedding (torch.Tensor): Node embeddings from the current GCN layer.
            previous_embedding (torch.Tensor): Node embeddings from the previous GCN layer (or initial embeddings).
            cas_indices (torch.Tensor): Indices of the cascades in the current batch.
            user_indices (torch.Tensor): Indices of the users in the current batch.

        Returns:
            torch.Tensor: The calculated SSL loss.
        """
        # Split embeddings into cascade and user parts
        current_cas_emb, current_user_emb = torch.split(current_embedding, [self.cas_num, self.user_num])
        prev_cas_emb_all, prev_user_emb_all = torch.split(previous_embedding, [self.cas_num, self.user_num])

        # --- User Contrastive Loss ---
        # Positive pairs: same user, different views (layers)
        user_emb1 = F.normalize(current_user_emb[user_indices])
        user_emb2 = F.normalize(prev_user_emb_all[user_indices])
        pos_score_user = torch.sum(user_emb1 * user_emb2, dim=1)

        # Negative pairs: current user vs. all users from the other view
        all_user_emb2 = F.normalize(prev_user_emb_all)
        ttl_score_user = torch.matmul(user_emb1, all_user_emb2.transpose(0, 1))

        # InfoNCE loss calculation for users
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # --- Cascade (Item) Contrastive Loss ---
        # Positive pairs: same cascade, different views (layers)
        item_emb1 = F.normalize(current_cas_emb[cas_indices])
        item_emb2 = F.normalize(prev_cas_emb_all[cas_indices])
        pos_score_item = torch.sum(item_emb1 * item_emb2, dim=1)

        # Negative pairs: current cascade vs. all cascades from the other view
        all_item_emb2 = F.normalize(prev_cas_emb_all)
        ttl_score_item = torch.matmul(item_emb1, all_item_emb2.transpose(0, 1))

        # InfoNCE loss calculation for cascades
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def proto_nce_loss(self, node_embedding, cas_indices, user_indices):
        """
        Computes the prototype-based contrastive loss (ProtoNCE).
        This loss encourages a node's embedding to be close to its assigned cluster centroid
        and far from other centroids.

        Args:
            node_embedding (torch.Tensor): The node embeddings (typically from the initial GCN layer).
            cas_indices (torch.Tensor): Indices of the cascades in the current batch.
            user_indices (torch.Tensor): Indices of the users in the current batch.

        Returns:
            torch.Tensor: The calculated prototype contrastive loss.
        """
        cas_emb_all, user_emb_all = torch.split(node_embedding, [self.cas_num, self.user_num])

        # --- User ProtoNCE Loss ---
        user_embeddings = F.normalize(user_emb_all[user_indices])
        user2cluster_indices = self.user_2cluster[user_indices]
        # Positive pairs: user embedding and its assigned centroid
        user_centroids = self.user_centroids[user2cluster_indices]
        pos_score_user = torch.sum(user_embeddings * user_centroids, dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        # Negative pairs: user embedding and all other centroids
        ttl_score_user = torch.matmul(user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        proto_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # --- Cascade (Item) ProtoNCE Loss ---
        cas_embeddings = F.normalize(cas_emb_all[cas_indices])
        cas2cluster_indices = self.cas_2cluster[cas_indices]
        # Positive pairs: cascade embedding and its assigned centroid
        cas_centroids = self.cas_centroids[cas2cluster_indices]
        pos_score_item = torch.sum(cas_embeddings * cas_centroids, dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        # Negative pairs: cascade embedding and all other centroids
        ttl_score_item = torch.matmul(cas_embeddings, self.cas_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_loss_user + proto_loss_item)
        return proto_nce_loss

    def forward(self, input_seq, input_timestamp, tgt_idx):
        """
        Defines the forward pass of the PMRCA model.

        Args:
            input_seq (torch.Tensor): The input sequence of user IDs, shape (batch_size, seq_len).
            input_timestamp (torch.Tensor): Timestamps for the input sequence (not used).
            tgt_idx (torch.Tensor): The target cascade indices for this batch.

        Returns:
            torch.Tensor or tuple:
                - If training: A tuple containing (predictions, user_embeddings, cascade_embeddings, embedding_list).
                - If not training: The raw prediction logits.
        """
        # Move input to the correct device
        input_seq = input_seq.to(self.device)

        # 1. Graph Convolution for Node Representation Enhancement
        ego_embeddings = self.get_ego_embeddings()
        embedding_list = [ego_embeddings]
        for _ in range(self.gcn_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            embedding_list.append(ego_embeddings)

        # Aggregate embeddings from all GCN layers
        gcn_all_embeddings = torch.stack(embedding_list, dim=1)
        gcn_all_embeddings = torch.mean(gcn_all_embeddings, dim=1)

        cas_all_embeddings, user_all_embeddings = torch.split(
            gcn_all_embeddings, [self.cas_num, self.user_num]
        )

        # 2. Sequence Modeling with Attention
        input_seq_for_pred = input_seq[:, :-1]
        
        # Create padding mask - mask padding positions
        # TransformerBlock expects a 2D mask [batch_size, seq_len]
        mask = (input_seq_for_pred == Constants.PAD)
        
        # Note: TransformerBlock internally handles causal masking
        # Causal masking is implemented in TransformerBlock.scaled_dot_product_attention

        # Add positional encodings to the dynamic embeddings
        batch_t = torch.arange(input_seq_for_pred.size(1)).expand_as(input_seq_for_pred).to(self.device)
        position_embed = self.pos_embedding(batch_t)
        dyemb = user_all_embeddings[input_seq_for_pred]
        dyemb += position_embed

        # 3. Intention Modeling
        original_seq_emb = self.user_embedding(input_seq_for_pred)
        # Pass mask to sequence encoder
        seq_out = self.seq_encoder(original_seq_emb, original_seq_emb, original_seq_emb, mask=mask)

        # 4. Alignment between dynamic embeddings and intention for prediction
        # Also pass mask to alignment attention mechanism for consistency
        att_out = self.align_attention(dyemb, seq_out, seq_out, mask=mask)

        # Final linear layer for prediction
        output = self.linear(att_out)

        # Mask users who have already appeared in the sequence
        prev_user_mask = self.get_previous_user_mask(input_seq_for_pred, self.user_num)
        output = output + prev_user_mask

        if self.training:
            return output.view(-1, output.size(-1)), user_all_embeddings, cas_all_embeddings, embedding_list
        else:
            return output.view(-1, output.size(-1))

    def get_previous_user_mask(self, seq, user_size):
        """
        Creates a mask to prevent the model from predicting users that are already in the input sequence.

        Args:
            seq (torch.Tensor): The input user sequence, shape (batch_size, seq_len).
            user_size (int): The total number of users.

        Returns:
            torch.Tensor: A mask tensor with large negative values at the positions of previous users.
        """
        assert seq.dim() == 2
        device = seq.device

        # Create a lower triangular matrix to mask future positions
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        previous_mask = torch.from_numpy(np.tril(np.ones(prev_shape), k=0)).to(device, dtype=torch.float32)

        # Repeat the sequence to align with the mask
        seqs = seq.unsqueeze(-1).expand(-1, -1, seq.size(1))

        # Apply the mask to get sequences of past users at each time step
        masked_seq = previous_mask * seqs.float()

        # Prepare for scattering: add a PAD dimension
        pad_tmp = torch.zeros(seq.size(0), seq.size(1), 1, device=device)
        masked_seq = torch.cat([masked_seq, pad_tmp], dim=2)

        # Create the final mask tensor by scattering -inf to the positions of previous users
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size, device=device)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))

        return Variable(masked_seq, requires_grad=False)

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

        pred, user_all_emb, cas_all_emb, embedding_list = self.forward(
            input_seq, input_seq_timestamp, history_seq_idx
        )

        center_embedding = embedding_list[0]
        # Using layer 1 as the context for contrastive loss, as per many SSL-GCN works.
        context_embedding = embedding_list[1]

        # 1. Multi-source Resonance (Inter-view Contrastive Loss)
        ssl_loss = self.ssl_layer_loss(
            context_embedding, center_embedding, history_seq_idx, gold[:, -1]
        )

        # 2. Cognitive Adaptation (Prototype-based Contrastive Loss)
        proto_loss = self.proto_nce_loss(center_embedding, history_seq_idx, gold[:, -1])

        # 3. Main task loss (Prediction)
        loss = loss_func(pred, gold.contiguous().view(-1))

        # Calculate accuracy
        pred_labels = pred.max(1)[1]
        gold_flat = gold.contiguous().view(-1)
        # Only consider non-padded elements for accuracy calculation
        non_pad_mask = gold_flat.ne(Constants.PAD).data
        n_correct = pred_labels.data.eq(gold_flat.data).masked_select(non_pad_mask).sum().float()

        # Combine all losses
        total_loss = loss + ssl_loss + proto_loss
        return total_loss, n_correct

    def before_epoch(self):
        """
        A hook that is called before the start of each training epoch.
        Used here to update the K-means clusters and centroids.
        """
        self.e_step()
