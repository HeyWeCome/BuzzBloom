import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from layers import GraphBuilder
import scipy.sparse as ss
from utils import Constants
from torch.autograd import Variable
from layers.Commons import DynamicGraphNN, GraphNN, Fusion, TimeAttention, ConvBlock
from layers.TransformerBlock import TransformerBlock


class DisenIDP(nn.Module):
    """
    Enhancing Information Diffusion Prediction with Self-Supervised Disentangled User and Cascade Representations
    Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM), 2023
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--pos_dim', type=int, default=8,
                            help='Dimension for positional encoding, in the original implement is 8.')
        parser.add_argument('--layer', type=float, default=3,
                            help='the number of layer used')
        parser.add_argument('--beta', type=float, default=0.001,
                            help='ssl graph task maginitude')
        parser.add_argument('--beta2', type=float, default=0.005,
                            help='ssl cascade task maginitude')
        return parser

    def __init__(self, args, data_loader):
        super(DisenIDP, self).__init__()
        self.n_node = data_loader.user_num
        self.emb_size = args.d_model
        self.layers = args.layer
        self.dropout = nn.Dropout(args.dropout)
        self.drop_rate = args.dropout
        self.win_size = 5
        self.beta = args.beta
        self.beta2 = args.beta2

        self.hyper_graphs = self._con_hyper_graph(self.n_node,
                                                  data_loader.train_data[0],
                                                  data_loader.timestamps,
                                                  self.win_size)
        self.n_channel = len(self.hyper_graphs)
        # hypergraph
        self.H_Item = self.hyper_graphs[0]
        self.H_User = self.hyper_graphs[1]

        # user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)
        # channel self-gating parameters
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        # channel self-supervised parameters
        self.ssl_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.ssl_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        # attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        # sequence model
        self.past_gru = nn.GRU(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)
        self.past_lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)

        # multi-head attention
        self.past_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=self.drop_rate)

        self.future_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=self.drop_rate)

        self.long_term_att = UnifiedAttention(input_size=self.emb_size, attn_dropout=self.drop_rate, long_term=True)
        self.short_term_att = UnifiedAttention(input_size=self.emb_size, attn_dropout=self.drop_rate)
        self.conv = ConvBlock(n_inputs=self.emb_size,
                              n_outputs=self.emb_size,
                              kernel_size=self.win_size,
                              padding=self.win_size - 1)
        self.linear = nn.Linear(self.emb_size * 3, self.emb_size)

        self.reset_parameters()

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def self_supervised_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.ssl_weights[channel]) + self.ssl_bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def hierarchical_ssl(self, em, adj):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        user_embeddings = em.cuda()
        edge_embeddings = torch.sparse.mm(adj.cuda(), em.cuda()).cuda()

        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        # Global MIM
        graph = torch.mean(edge_embeddings, 0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def seq2seq_ssl(self, L_fea1, L_fea2, S_fea1, S_fea2):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), -1)

        # Local MIM
        pos = score(L_fea1, L_fea2)
        neg1 = score(L_fea1, S_fea2)
        neg2 = score(L_fea2, S_fea1)
        loss1 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        pos = score(S_fea1, S_fea2)
        loss2 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        return loss1 + loss2

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g

    '''social structure and hypergraph structure embeddding'''

    def structure_embed(self, H_Time=None, H_Item=None, H_User=None):

        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1 - self.drop_rate).cuda()
            H_User = self._dropout_graph(self.H_User, keep_prob=1 - self.drop_rate).cuda()
        else:
            H_Item = self.H_Item.cuda()
            H_User = self.H_User.cuda()

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)

        all_emb_c2 = [u_emb_c2]
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            # Channel Item
            u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)
            norm_embeddings2 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings2]

        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)
        u_emb_c3 = torch.stack(all_emb_c3, dim=1)
        u_emb_c3 = torch.sum(u_emb_c3, dim=1)

        # aggregating channel-specific embeddings
        high_embs, attention_score = self.channel_attention(u_emb_c2, u_emb_c3)

        return high_embs

    def forward(self, input, tgt_timestamp, tgt_idx, label=None):
        input = input[:, :-1]

        mask = (input == 0)
        mask_label = (label == 0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        # long-term temporal influence
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask=mask.cuda())

        # short-term temporal influence
        user_cas_gru, _ = self.past_gru(cas_seq_emb)
        user_cas_lstm, _ = self.past_lstm(cas_seq_emb)
        S_cas_emb = self.short_term_att(user_cas_gru, user_cas_lstm, user_cas_lstm, mask=mask.cuda())

        LS_cas_emb = torch.cat([cas_seq_emb, L_cas_emb, S_cas_emb], -1)
        LS_cas_emb = self.linear(LS_cas_emb)

        output = self.past_multi_att(LS_cas_emb, LS_cas_emb, LS_cas_emb, mask)

        if self.training:
            # future cascade
            future_embs = F.embedding(label, HG_Uemb)
            future_output = self.future_multi_att(future_embs, future_embs, future_embs, mask=mask_label.cuda())
            # future cascade
            short_emb = self.conv(cas_seq_emb)
            '''SSL loss'''
            graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 0), self.H_Item)
            graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 1), self.H_User)

            seq_ssl_loss = self.seq2seq_ssl(L_cas_emb, future_output, S_cas_emb, short_emb)

            '''Prediction'''
            pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
            mask = self.get_previous_user_mask(input, self.n_node)
            return (pre_y + mask).view(-1, pre_y.size(-1)).cuda(), graph_ssl_loss, seq_ssl_loss
        else:
            '''Prediction'''
            pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
            mask = self.get_previous_user_mask(input, self.n_node)
            return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()

    def get_previous_user_mask(self, seq, user_size):
        """ Mask previous activated users."""
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()
        masked_seq = previous_mask * seqs.data.float()

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
        masked_seq = Variable(masked_seq, requires_grad=False)
        # print("masked_seq ",masked_seq.size())
        return masked_seq.cuda()

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        if self.training:
            pred, graph_ssl_loss, seq_ssl_loss = self.forward(input_seq, input_seq_timestamp, history_seq_idx, gold)
            loss = loss_func(pred, gold.contiguous().view(-1))
            pred = pred.max(1)[1]
            gold = gold.contiguous().view(-1)
            n_correct = pred.data.eq(gold.data)
            n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

            final_loss = loss + self.beta * graph_ssl_loss + self.beta2 * seq_ssl_loss
            return final_loss, n_correct
        else:
            pred = self.forward(input_seq, gold)
            loss = loss_func(pred, gold.contiguous().view(-1))
            pred = pred.max(1)[1]
            gold = gold.contiguous().view(-1)
            n_correct = pred.data.eq(gold.data)
            n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
            return loss, n_correct

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    '''Hypergraph'''

    def _con_hyper_graph(self, user_size, all_cascade, all_time, window):
        ###context
        user_cont = {}
        for i in range(user_size):
            user_cont[i] = []

        win = window
        for i in range(len(all_cascade)):
            cas = all_cascade[i]

            if len(cas) < win:
                for idx in cas:
                    user_cont[idx] = list(set(user_cont[idx] + cas))
                continue
            for j in range(len(cas) - win + 1):
                if (j + win) > len(cas):
                    break
                cas_win = cas[j:j + win]
                for idx in cas_win:
                    user_cont[idx] = list(set(user_cont[idx] + cas_win))

        indptr, indices, data = [], [], []
        indptr.append(0)
        idx = 0

        for j in user_cont.keys():

            # idx = source_users[j]
            if len(user_cont[j]) == 0:
                idx = idx + 1
                continue
            source = np.unique(user_cont[j])

            length = len(source)
            s = indptr[-1]
            indptr.append((s + length))
            for i in range(length):
                indices.append(source[i])
                data.append(1)

        H_U = ss.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys()) - idx, user_size))

        H_U_sum = 1.0 / H_U.sum(axis=1).reshape(1, -1)
        H_U_sum[H_U_sum == float("inf")] = 0

        # BH_T = H_S.T.multiply(1.0 / H_S.sum(axis=1).reshape(1, -1))
        BH_T = H_U.T.multiply(H_U_sum)
        BH_T = BH_T.T
        H = H_U.T

        epsilon = 1e-10  # Small constant to prevent division by zero
        H_sum = 1.0 / (H.sum(axis=1).reshape(1, -1) + epsilon)
        H_sum[H_sum == float("inf")] = 0

        DH = H.T.multiply(H_sum)
        # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        HG_User = np.dot(DH, BH_T).tocoo()

        '''U-I hypergraph'''
        indptr, indices, data = [], [], []
        indptr.append(0)
        for j in range(len(all_cascade)):
            items = np.unique(all_cascade[j])

            length = len(items)

            s = indptr[-1]
            indptr.append((s + length))
            for i in range(length):
                indices.append(items[i])
                data.append(1)

        H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))

        H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
        H_T_sum[H_T_sum == float("inf")] = 0

        # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = H_T.T.multiply(H_T_sum)
        BH_T = BH_T.T
        H = H_T.T

        H_sum = 1.0 / (H.sum(axis=1).reshape(1, -1) + epsilon)
        H_sum[H_sum == float("inf")] = 0

        DH = H.T.multiply(H_sum)
        # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        HG_Item = np.dot(DH, BH_T).tocoo()

        HG_Item = self._convert_sp_mat_to_sp_tensor(HG_Item)
        HG_User = self._convert_sp_mat_to_sp_tensor(HG_User)

        return HG_Item, HG_User


class UnifiedAttention(nn.Module):
    def __init__(self, input_size, attn_dropout=0.1, long_term=False):
        super(UnifiedAttention, self).__init__()

        self.input_size = input_size
        self.long_term = long_term  # 控制是否使用 Long_term_atention 的特性
        self.W_q = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size, input_size))

        self.dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(input_size)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.normal_(self.W_q, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_k, mean=0, std=np.sqrt(2.0 / (self.input_size)))
        nn.init.normal_(self.W_v, mean=0, std=np.sqrt(2.0 / (self.input_size)))

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        temperature = self.input_size ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)

        # 控制是否重复 Q_K，类似于 Long_term_atention 的处理方式
        if self.long_term:
            _, k_len, _ = K.size()
            Q_K = Q_K.repeat(1, k_len, 1)

        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            upper_triangle_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
            mask_ = upper_triangle_mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -1e10)

        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V, mask):
        Q_ = Q.matmul(self.W_q)
        K_ = K.matmul(self.W_k)
        V_ = V.matmul(self.W_v)
        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        return V_att

    def forward(self, Q, K, V, mask=None):
        # 根据 long_term 参数决定是否扩展 Q 的维度
        if self.long_term:
            Q = Q.unsqueeze(dim=1)

        V_att = self.multi_head_attention(Q, K, V, mask)
        output = self.layer_norm(V + V_att)
        return output
