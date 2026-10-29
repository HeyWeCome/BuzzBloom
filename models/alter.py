"""Core implementation of ALTER.

The implementation follows Algorithm 1 and the Method section of the paper:

* Adaptive Latent Transport: Eqs. (6)--(12).
* Contextual Message Refinement: Eqs. (13)--(14).
* Cascade Encoding and Diffusion Prediction: Eq. (3) and its surrounding text.
* Training Objective: Eqs. (4), (15), and (16).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model, requires_content_user_graph
from models.modules.graph import build_normalized_content_user_adj, stable_sparse_mm
from models.modules.sequence_encoder import TransformerEncoderLayer


def _top_m_softmax(logits: torch.Tensor, budget: int, dim: int = -1) -> torch.Tensor:
    """Form a dense or top M normalized assignment matrix.

    This implements the assignment normalization in Eqs. (7) and (10),
    labeled ``eq:source_assignment`` and ``eq:recipient_assignment``.
    Nonselected entries are exactly zero when the budget is active.
    """
    if budget <= 0 or budget >= logits.size(dim):
        return F.softmax(logits, dim=dim)

    selected_values, selected_indices = torch.topk(logits, k=budget, dim=dim)
    selected_weights = F.softmax(selected_values, dim=dim)
    weights = torch.zeros_like(logits)
    return weights.scatter(dim, selected_indices, selected_weights)


class _TwoStreamMixer(nn.Module):
    """Mix two message streams as defined by Eqs. (11) and (12).

    The same form is used for observed and transported messages and, with
    separate parameters, for fused messages and contextual references.
    See ``eq:transport_gate`` and ``eq:transport_mix``.
    """

    def __init__(self, d_model: int, temperature: float) -> None:
        super().__init__()
        self.left_score = nn.Linear(d_model, 1)
        self.right_score = nn.Linear(d_model, 1)
        self.mix = nn.Linear(2, 2)
        self.temperature = max(float(temperature), 1e-6)

    def reset_parameters(self) -> None:
        for module in [self.left_score, self.right_score, self.mix]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        scores = torch.cat(
            [
                torch.sigmoid(self.left_score(left)),
                torch.sigmoid(self.right_score(right)),
            ],
            dim=-1,
        )
        weights = F.softmax(self.mix(scores / self.temperature), dim=-1)
        return weights[:, :1] * left + weights[:, 1:] * right


class _ALTERLayer(nn.Module):
    """One graph layer from the loop in Algorithm 1.

    The layer retains observed propagation while performing source assignment,
    transport state updating, recipient assignment, message fusion, contextual
    refinement, and the residual node update in Eqs. (6)--(14).
    """

    def __init__(
        self,
        d_model: int,
        num_context_states: int,
        dropout: float,
        temperature: float,
        assignment_budget: int,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.assignment_budget = int(assignment_budget)

        self.local_projection = nn.Linear(d_model, d_model)
        self.transport_state_update = nn.Linear(d_model, d_model)
        self.transport_projection = nn.Linear(d_model, d_model)
        self.transport_state_norm = nn.LayerNorm(d_model)

        self.transport_mixer = _TwoStreamMixer(d_model, temperature)
        self.context_bank = nn.Parameter(torch.empty(num_context_states, d_model))
        self.context_mixer = _TwoStreamMixer(d_model, temperature)

        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        """Use the initialization stated in Supplement, Implementation Details."""
        for module in [
            self.local_projection,
            self.transport_state_update,
            self.transport_projection,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.context_bank)
        self.transport_mixer.reset_parameters()
        self.context_mixer.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        normalized_adjacency: torch.Tensor,
        routing_anchors: torch.Tensor,
        valid_node_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Observed propagation in Sec. Adaptive Latent Transport, Eq. (6)
        # (``eq:normalized_adj``) and the definition of M_o immediately after it.
        observed_message = stable_sparse_mm(normalized_adjacency, node_states)
        observed_message = torch.nan_to_num(observed_message, nan=0.0)
        observed_message = F.gelu(self.local_projection(observed_message))

        # Sparse source assignment R^(r), Eq. (7)
        # (``eq:source_assignment``). Invalid special nodes do not act as sources.
        normalized_nodes = F.normalize(node_states, p=2, dim=-1)
        normalized_anchors = F.normalize(routing_anchors, p=2, dim=-1)
        source_affinity = normalized_nodes @ normalized_anchors.t()
        source_assignment = _top_m_softmax(
            source_affinity / self.temperature,
            self.assignment_budget,
            dim=-1,
        )
        source_assignment = source_assignment * valid_node_mask.to(
            device=node_states.device,
            dtype=source_assignment.dtype,
        ).unsqueeze(-1)

        # Source aggregation and source conditioned transport state update,
        # Eqs. (8)--(9) (``eq:source_aggregation`` and
        # ``eq:transport_state_update``).
        source_mass = source_assignment.sum(dim=0).clamp_min(1e-12)
        source_aggregate = source_assignment.t() @ node_states
        source_aggregate = source_aggregate / source_mass.unsqueeze(-1)
        transport_delta = F.gelu(self.transport_state_update(source_aggregate))
        transport_states = self.transport_state_norm(routing_anchors + transport_delta)

        # Recipient assignment B^(r) is evaluated after the transport state
        # update, as required by Eq. (10) (``eq:recipient_assignment``).
        recipient_affinity = normalized_nodes @ F.normalize(
            transport_states,
            p=2,
            dim=-1,
        ).t()
        recipient_assignment = _top_m_softmax(
            recipient_affinity / self.temperature,
            self.assignment_budget,
            dim=-1,
        )
        transported_message = recipient_assignment @ transport_states
        transported_message = torch.nan_to_num(transported_message, nan=0.0)
        transported_message = F.gelu(self.transport_projection(transported_message))

        # Fuse the observed and transported branches using Eqs. (11)--(12)
        # (``eq:transport_gate`` and ``eq:transport_mix``).
        fused_message = self.transport_mixer(
            observed_message,
            transported_message,
        )

        # Construct the contextual reference M_c^(r), Eq. (13)
        # (``eq:context_aggregation``).
        context_affinity = (
            F.normalize(fused_message, p=2, dim=-1)
            @ F.normalize(self.context_bank, p=2, dim=-1).t()
        )
        context_weights = F.softmax(context_affinity, dim=-1)
        contextual_reference = context_weights @ self.context_bank
        refined_message = self.context_mixer(
            fused_message,
            contextual_reference,
        )

        # Nodewise residual update, Eq. (14) (``eq:node_update``).
        return self.output_norm(
            node_states + self.dropout(F.gelu(refined_message))
        )


@register_model("alter")
@requires_content_user_graph
class ALTER(nn.Module):
    """Adaptive latent transport for information diffusion prediction.

    ``_ALTERLayer`` implements the graph loop in Algorithm 1. The remaining
    methods implement the layer average, cascade encoder, candidate scoring,
    and training objective described after that loop in the paper.
    """

    @staticmethod
    def add_model_args(parser) -> None:
        group = parser.add_argument_group("ALTER")
        group.add_argument("--alter_layers", type=int, default=3)
        group.add_argument("--alter_routing_anchors", type=int, default=16)
        group.add_argument("--alter_context_states", type=int, default=16)
        group.add_argument("--alter_temperature", type=float, default=2.0)
        group.add_argument("--alter_assignment_budget", type=int, default=2)
        group.add_argument(
            "--alter_latent_regularization_weight",
            type=float,
            default=1e-6,
        )

    @staticmethod
    def build_model_kwargs(args, meta) -> dict:
        del meta
        return {
            "graph_layers": args.alter_layers,
            "num_routing_anchors": args.alter_routing_anchors,
            "num_context_states": args.alter_context_states,
            "assignment_temperature": args.alter_temperature,
            "assignment_budget": args.alter_assignment_budget,
            "latent_regularization_weight": args.alter_latent_regularization_weight,
        }

    @staticmethod
    def run_name_parts(args):
        budget = (
            "dense"
            if args.alter_assignment_budget <= 0
            else args.alter_assignment_budget
        )
        return [
            ("gl", args.alter_layers),
            ("k", args.alter_routing_anchors),
            ("kc", args.alter_context_states),
            ("m", budget),
            ("tau", args.alter_temperature),
            ("lreg", args.alter_latent_regularization_weight),
        ]

    def __init__(
        self,
        num_users: int,
        num_actions: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        max_time_bucket: int = 32,
        normalize_embeddings: bool = False,
        logit_scale: float = 1.0,
        learnable_scale: bool = False,
        num_contents: int = 0,
        train_content_user_edges: Optional[list[tuple[int, int]]] = None,
        graph_layers: int = 2,
        num_routing_anchors: int = 64,
        num_context_states: int = 64,
        assignment_temperature: float = 2.0,
        assignment_budget: int = 2,
        latent_regularization_weight: float = 1e-6,
    ) -> None:
        super().__init__()
        del num_actions
        if num_contents <= 0:
            raise ValueError(
                "ALTER requires content nodes from the training participation graph."
            )
        if train_content_user_edges is None:
            raise ValueError(
                "ALTER requires edges from the training participation graph."
            )

        self.num_users = int(num_users)
        self.num_contents = int(num_contents)
        self.num_graph_nodes = self.num_contents + self.num_users
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.graph_layer_count = max(int(graph_layers), 1)
        self.num_routing_anchors = max(int(num_routing_anchors), 1)
        self.num_context_states = max(int(num_context_states), 1)
        self.assignment_budget = int(assignment_budget)
        self.latent_regularization_weight = float(latent_regularization_weight)

        self.user_embedding = nn.Embedding(self.num_users, d_model, padding_idx=0)
        self.raw_user_embedding = nn.Embedding(
            self.num_users,
            d_model,
            padding_idx=0,
        )
        self.age_embedding = nn.Embedding(
            max_time_bucket + 1,
            d_model,
            padding_idx=0,
        )
        self.content_embedding = nn.Embedding(self.num_contents, d_model)
        self.routing_anchors = nn.Parameter(
            torch.empty(self.num_routing_anchors, d_model)
        )

        self.transport_layers = nn.ModuleList(
            [
                _ALTERLayer(
                    d_model=d_model,
                    num_context_states=self.num_context_states,
                    dropout=dropout,
                    temperature=assignment_temperature,
                    assignment_budget=self.assignment_budget,
                )
                for _ in range(self.graph_layer_count)
            ]
        )

        self.user_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.user_fusion_norm = nn.LayerNorm(d_model)

        self.cascade_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_mult=4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        head_dim = d_model // num_heads
        theta = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, theta)
        frequencies = torch.cat([frequencies, frequencies], dim=-1)
        self.register_buffer(
            "rope_cos_cached",
            frequencies.cos().unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin_cached",
            frequencies.sin().unsqueeze(0),
            persistent=False,
        )

        normalized_adjacency = build_normalized_content_user_adj(
            num_contents=self.num_contents,
            num_users=self.num_users,
            edges=train_content_user_edges,
        )
        # The fixed normalized observed graph is the matrix in Eq. (6),
        # ``eq:normalized_adj``.
        self.register_buffer("normalized_adjacency", normalized_adjacency)

        # Padding and unknown user nodes are excluded from source aggregation,
        # following Supplement, Implementation Details.
        valid_node_mask = torch.ones(self.num_graph_nodes, dtype=torch.bool)
        valid_node_mask[self.num_contents] = False
        if self.num_users > 1:
            valid_node_mask[self.num_contents + 1] = False
        self.register_buffer("valid_node_mask", valid_node_mask)

        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(logit_scale))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings and latent states as stated in the Supplement."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.raw_user_embedding.weight)
        nn.init.xavier_uniform_(self.age_embedding.weight)
        nn.init.xavier_uniform_(self.content_embedding.weight)
        nn.init.xavier_uniform_(self.routing_anchors)
        for layer in self.transport_layers:
            layer.reset_parameters()
        for module in self.user_fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.raw_user_embedding.weight[0].zero_()
            self.age_embedding.weight[0].zero_()

    def _get_rope_tables(
        self,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = min(sequence_length, self.max_seq_len)
        return (
            self.rope_cos_cached[:, :length, :],
            self.rope_sin_cached[:, :length, :],
        )

    def _initial_graph_states(self) -> torch.Tensor:
        """Build H^(0) from content and user embeddings.

        This corresponds to the initial graph state defined at the beginning of
        the Method section. The row order follows the graph builder convention:
        content nodes precede user nodes.
        """
        return torch.cat(
            [self.content_embedding.weight, self.user_embedding.weight],
            dim=0,
        )

    def _compute_graph_embeddings(self) -> dict[str, torch.Tensor]:
        """Run Algorithm 1 over graph layers and average their representations.

        The final average implements Z = 1/(L+1) sum_r H^(r) in Sec. Cascade
        Encoding and Diffusion Prediction.
        """
        initial_states = self._initial_graph_states()
        device_type = initial_states.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            current_states = initial_states.float()
            normalized_adjacency = self.normalized_adjacency.float()
            routing_anchors = self.routing_anchors.float()
            layer_outputs = [current_states]
            pad_user_node = self.num_contents

            for layer in self.transport_layers:
                current_states = layer(
                    current_states,
                    normalized_adjacency,
                    routing_anchors,
                    self.valid_node_mask,
                )
                current_states = torch.nan_to_num(current_states, nan=0.0)
                current_states = current_states.clone()
                current_states[pad_user_node].zero_()
                layer_outputs.append(current_states)

            averaged_states = torch.mean(
                torch.stack(layer_outputs, dim=1),
                dim=1,
            )
            content_states, enhanced_user_states = torch.split(
                averaged_states,
                [self.num_contents, self.num_users],
            )
            enhanced_user_states = enhanced_user_states.clone()
            enhanced_user_states[0].zero_()

        return {
            "content_states": content_states.to(dtype=initial_states.dtype),
            "enhanced_user_states": enhanced_user_states.to(
                dtype=initial_states.dtype
            ),
        }

    def _fuse_user_embeddings(
        self,
        enhanced_user_states: torch.Tensor,
    ) -> torch.Tensor:
        """Combine graph states with independent user embeddings.

        This implements z_u = LN(psi_f([z_u^g; e_u])) in Sec. Cascade Encoding
        and Diffusion Prediction.
        """
        raw_user_embeddings = self.raw_user_embedding.weight.to(
            device=enhanced_user_states.device,
            dtype=enhanced_user_states.dtype,
        )
        user_embeddings = self.user_fusion(
            torch.cat([enhanced_user_states, raw_user_embeddings], dim=-1)
        )
        user_embeddings = self.user_fusion_norm(user_embeddings)
        user_embeddings = user_embeddings.clone()
        user_embeddings[0].zero_()
        return user_embeddings

    @staticmethod
    def _last_valid_state(
        sequence_states: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        lengths = (~key_padding_mask).sum(dim=1).clamp(min=1)
        last_indices = lengths - 1
        row_indices = torch.arange(
            sequence_states.size(0),
            device=sequence_states.device,
        )
        return sequence_states[row_indices, last_indices]

    def encode_prefix(
        self,
        prefix_users: torch.Tensor,
        prefix_actions: torch.Tensor,
        event_age_buckets: torch.Tensor,
        inter_delta_buckets: torch.Tensor,
        key_padding_mask: torch.Tensor,
        return_graph: bool = False,
    ):
        """Encode a cascade prefix into h_(m,t).

        This follows Sec. Cascade Encoding and Diffusion Prediction: event age
        embeddings are added to user representations, rotary positional
        encoding is applied by the Transformer, and the last valid state is
        retained as the cascade representation.
        """
        del prefix_actions, inter_delta_buckets
        graph = self._compute_graph_embeddings()
        user_embeddings = self._fuse_user_embeddings(
            graph["enhanced_user_states"]
        )
        graph["user_embeddings"] = user_embeddings

        sequence_states = (
            user_embeddings[prefix_users]
            + self.age_embedding(event_age_buckets)
        )
        rope_cos, rope_sin = self._get_rope_tables(sequence_states.size(1))
        for layer in self.cascade_layers:
            sequence_states, updated_mask = layer(
                sequence_states,
                key_padding_mask=key_padding_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            if updated_mask is not None:
                key_padding_mask = updated_mask

        cascade_state = self._last_valid_state(
            sequence_states,
            key_padding_mask,
        )
        if return_graph:
            return cascade_state, graph
        return cascade_state

    def _score_candidates(
        self,
        cascade_state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute candidate logits for Eq. (3), ``eq:next_user_probability``."""
        if self.normalize_embeddings:
            cascade_state = F.normalize(cascade_state, p=2, dim=-1)
            candidate_embeddings = F.normalize(
                candidate_embeddings,
                p=2,
                dim=-1,
            )
        scores = torch.bmm(
            candidate_embeddings,
            cascade_state.unsqueeze(-1),
        ).squeeze(-1)
        return scores * self.logit_scale

    def forward(
        self,
        prefix_users: torch.Tensor,
        prefix_actions: torch.Tensor,
        event_age_buckets: torch.Tensor,
        inter_delta_buckets: torch.Tensor,
        candidate_users: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        cascade_state, graph = self.encode_prefix(
            prefix_users,
            prefix_actions,
            event_age_buckets,
            inter_delta_buckets,
            key_padding_mask,
            return_graph=True,
        )
        return self._score_candidates(
            cascade_state,
            graph["user_embeddings"][candidate_users],
        )

    def score_all_train_users(
        self,
        prefix_users: torch.Tensor,
        prefix_actions: torch.Tensor,
        event_age_buckets: torch.Tensor,
        inter_delta_buckets: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        cascade_state, graph = self.encode_prefix(
            prefix_users,
            prefix_actions,
            event_age_buckets,
            inter_delta_buckets,
            key_padding_mask,
            return_graph=True,
        )
        candidate_embeddings = graph["user_embeddings"][2:]
        if self.normalize_embeddings:
            cascade_state = F.normalize(cascade_state, p=2, dim=-1)
            candidate_embeddings = F.normalize(
                candidate_embeddings,
                p=2,
                dim=-1,
            )
        return (
            cascade_state @ candidate_embeddings.t()
        ) * self.logit_scale

    def _sample_negatives(
        self,
        target_user: torch.Tensor,
        prefix_users: torch.Tensor,
        key_padding_mask: torch.Tensor,
        num_negatives: int,
    ) -> torch.Tensor:
        """Sample training candidates under the paper evaluation protocol.

        The true target and users already observed in the prefix are excluded,
        as specified in Experimental Details and the Technical Supplement.
        """
        batch_size = target_user.shape[0]
        device = target_user.device
        train_users = torch.arange(
            2,
            self.num_users,
            device=device,
            dtype=torch.long,
        )
        pool_size = train_users.shape[0]
        if pool_size <= 0:
            raise RuntimeError(
                "ALTER needs at least one non special user for negative sampling."
            )

        weights = torch.ones(
            batch_size,
            pool_size,
            dtype=torch.float32,
            device=device,
        )
        target_indices = target_user - 2
        target_valid = (
            (target_indices >= 0)
            & (target_indices < pool_size)
        )
        rows = torch.arange(batch_size, device=device)
        weights[rows[target_valid], target_indices[target_valid]] = 0.0

        if prefix_users.numel() > 0:
            prefix_indices = prefix_users - 2
            prefix_valid = (
                (~key_padding_mask)
                & (prefix_indices >= 0)
                & (prefix_indices < pool_size)
            )
            row_indices = rows.unsqueeze(1).expand_as(prefix_indices)
            weights[
                row_indices[prefix_valid],
                prefix_indices[prefix_valid],
            ] = 0.0

        replacement = bool(
            torch.any(
                (weights > 0).sum(dim=1) < num_negatives
            ).item()
        )
        sampled_indices = torch.multinomial(
            weights,
            num_samples=num_negatives,
            replacement=replacement,
        )
        return train_users[sampled_indices]

    @staticmethod
    def _latent_parameter_penalty(parameters: torch.Tensor) -> torch.Tensor:
        """Compute the normalized squared L2 and L1 penalty in Eq. (15)."""
        return parameters.pow(2).mean() + parameters.abs().mean()

    def _latent_regularization(self) -> torch.Tensor:
        """Apply Eq. (15) to anchors and context banks as required by Eq. (16)."""
        loss = self.user_embedding.weight.new_zeros(())
        if self.latent_regularization_weight <= 0:
            return loss

        penalty = self._latent_parameter_penalty(self.routing_anchors)
        for layer in self.transport_layers:
            penalty = penalty + self._latent_parameter_penalty(
                layer.context_bank
            )
        return self.latent_regularization_weight * penalty

    def compute_loss(
        self,
        batch: dict,
        num_negatives: int = 100,
    ) -> torch.Tensor:
        """Compute the sampled task loss and latent regularization.

        Cross entropy over the sampled candidate set implements the task term
        in Eq. (4), and the returned sum follows Eq. (16),
        ``eq:overall_loss``.
        """
        prefix_users = batch["prefix_users"]
        key_padding_mask = batch["key_padding_mask"]
        target_user = batch["target_user"]

        cascade_state, graph = self.encode_prefix(
            prefix_users,
            batch["prefix_actions"],
            batch["event_age_buckets"],
            batch["inter_delta_buckets"],
            key_padding_mask,
            return_graph=True,
        )
        negative_users = self._sample_negatives(
            target_user,
            prefix_users,
            key_padding_mask,
            num_negatives,
        )
        candidate_users = torch.cat(
            [target_user.unsqueeze(1), negative_users],
            dim=1,
        )
        scores = self._score_candidates(
            cascade_state,
            graph["user_embeddings"][candidate_users],
        )
        task_loss = F.cross_entropy(
            scores,
            torch.zeros(
                scores.size(0),
                dtype=torch.long,
                device=scores.device,
            ),
        )
        return task_loss + self._latent_regularization()
