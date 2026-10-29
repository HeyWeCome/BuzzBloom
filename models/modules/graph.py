from __future__ import annotations

import torch


DEFAULT_MAX_SPARSE_MESSAGE_BYTES = 256 * 1024 * 1024


def stable_sparse_mm(
    adj: torch.Tensor,
    dense: torch.Tensor,
    max_message_bytes: int = DEFAULT_MAX_SPARSE_MESSAGE_BYTES,
) -> torch.Tensor:
    if dense.dim() != 2:
        raise ValueError(f"stable_sparse_mm expects a 2D dense matrix, got shape={tuple(dense.shape)}")

    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    if values.numel() == 0:
        return dense.new_zeros((adj.size(0), dense.size(1)))

    row = indices[0]
    col = indices[1]
    lengths = torch.bincount(row, minlength=adj.size(0))
    message_bytes = values.numel() * dense.size(1) * dense.element_size()
    if max_message_bytes <= 0 or message_bytes <= max_message_bytes:
        messages = dense.index_select(0, col) * values.to(dtype=dense.dtype).unsqueeze(-1)
        return torch.segment_reduce(messages, "sum", lengths=lengths, axis=0)

    bytes_per_edge = max(dense.size(1) * dense.element_size(), 1)
    max_edges = max(int(max_message_bytes // bytes_per_edge), 1)
    row_offsets = torch.empty(adj.size(0) + 1, dtype=torch.long, device=row.device)
    row_offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=row_offsets[1:])

    out = dense.new_zeros((adj.size(0), dense.size(1)))
    start_row = 0
    total_edges = values.numel()
    while start_row < adj.size(0):
        edge_start = int(row_offsets[start_row].item())
        if edge_start >= total_edges:
            break

        target_edge_end = min(edge_start + max_edges, total_edges)
        end_row = int(torch.searchsorted(row_offsets, target_edge_end, right=False).item())
        if end_row <= start_row:
            end_row = start_row + 1

        edge_end = int(row_offsets[end_row].item())
        chunk_messages = (
            dense.index_select(0, col[edge_start:edge_end])
            * values[edge_start:edge_end].to(dtype=dense.dtype).unsqueeze(-1)
        )
        out[start_row:end_row] = torch.segment_reduce(
            chunk_messages,
            "sum",
            lengths=lengths[start_row:end_row],
            axis=0,
        )
        start_row = end_row
    return out


def stable_index_sum(index: torch.Tensor, values: torch.Tensor, dim_size: int) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"stable_index_sum expects 2D values, got shape={tuple(values.shape)}")
    if index.numel() != values.size(0):
        raise ValueError("stable_index_sum index and values must have the same first dimension.")
    if index.numel() == 0:
        return values.new_zeros((dim_size, values.size(1)))

    order = torch.argsort(index)
    sorted_index = index[order]
    sorted_values = values.index_select(0, order)
    lengths = torch.bincount(sorted_index, minlength=dim_size)
    return torch.segment_reduce(sorted_values, "sum", lengths=lengths, axis=0)


def build_content_user_adj(
    num_contents: int,
    num_users: int,
    edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> torch.Tensor:
    if num_contents <= 0:
        raise ValueError("num_contents must be positive.")
    if num_users <= 0:
        raise ValueError("num_users must be positive.")

    total_nodes = num_contents + num_users
    if not edges:
        empty_indices = torch.empty((2, 0), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            empty_indices,
            empty_values,
            size=(total_nodes, total_nodes),
            dtype=torch.float32,
        ).coalesce()

    src: list[int] = []
    dst: list[int] = []
    for content_idx, user_idx in edges:
        if not (0 <= content_idx < num_contents and 0 <= user_idx < num_users):
            continue
        user_node = num_contents + user_idx
        src.extend([content_idx, user_node])
        dst.extend([user_node, content_idx])

    indices = torch.tensor([src, dst], dtype=torch.long)
    values = torch.ones(len(src), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(total_nodes, total_nodes),
        dtype=torch.float32,
    ).coalesce()


def normalize_sparse_adj(adj: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    degree = torch.zeros(adj.size(0), dtype=values.dtype, device=values.device)
    degree.scatter_add_(0, indices[0], values)
    deg_inv_sqrt = degree.add(eps).pow(-0.5)
    norm_values = values * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices,
        norm_values,
        size=adj.shape,
        dtype=values.dtype,
        device=values.device,
    ).coalesce()


def build_normalized_content_user_adj(
    num_contents: int,
    num_users: int,
    edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> torch.Tensor:
    return normalize_sparse_adj(build_content_user_adj(num_contents, num_users, edges))


def build_temporal_normalized_content_user_adjs(
    num_contents: int,
    num_users: int,
    events: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...],
    time_splits: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if time_splits <= 0:
        raise ValueError("time_splits must be positive.")

    if not events:
        empty_adj = build_normalized_content_user_adj(num_contents, num_users, [])
        return torch.zeros(1, dtype=torch.long), [empty_adj]

    sorted_events = sorted(events, key=lambda item: item[3])
    num_snapshots = min(time_splits, len(sorted_events))
    cutoff_positions: list[int] = []
    for split_idx in range(1, num_snapshots + 1):
        cutoff_pos = (split_idx * len(sorted_events) + num_snapshots - 1) // num_snapshots - 1
        cutoff_positions.append(min(max(cutoff_pos, 0), len(sorted_events) - 1))

    snapshot_orders: list[int] = []
    snapshot_adjs: list[torch.Tensor] = []
    edge_cursor = 0
    active_edges: list[tuple[int, int]] = []

    for cutoff_pos in cutoff_positions:
        cutoff_order = sorted_events[cutoff_pos][3]
        while edge_cursor < len(sorted_events) and sorted_events[edge_cursor][3] <= cutoff_order:
            content_idx, user_idx, _, _ = sorted_events[edge_cursor]
            active_edges.append((content_idx, user_idx))
            edge_cursor += 1

        if snapshot_orders and snapshot_orders[-1] == cutoff_order:
            snapshot_adjs[-1] = build_normalized_content_user_adj(num_contents, num_users, active_edges)
            continue

        snapshot_orders.append(cutoff_order)
        snapshot_adjs.append(build_normalized_content_user_adj(num_contents, num_users, active_edges))

    return torch.tensor(snapshot_orders, dtype=torch.long), snapshot_adjs


def _normalized_incidence(
    num_contents: int,
    num_users: int,
    edges: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not edges:
        empty_user_content = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=(num_users, num_contents),
            dtype=torch.float32,
        ).coalesce()
        empty_content_user = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=(num_contents, num_users),
            dtype=torch.float32,
        ).coalesce()
        return empty_user_content, empty_content_user

    user_idx = torch.tensor([user for _, user in edges], dtype=torch.long)
    content_idx = torch.tensor([content for content, _ in edges], dtype=torch.long)

    user_degree = torch.bincount(user_idx, minlength=num_users).float().clamp(min=1.0)
    content_degree = torch.bincount(content_idx, minlength=num_contents).float().clamp(min=1.0)

    user_content = torch.sparse_coo_tensor(
        torch.stack([user_idx, content_idx]),
        user_degree[user_idx].reciprocal(),
        size=(num_users, num_contents),
        dtype=torch.float32,
    ).coalesce()
    content_user = torch.sparse_coo_tensor(
        torch.stack([content_idx, user_idx]),
        content_degree[content_idx].reciprocal(),
        size=(num_contents, num_users),
        dtype=torch.float32,
    ).coalesce()
    return user_content, content_user


def build_temporal_content_user_incidence_matrices(
    num_contents: int,
    num_users: int,
    events: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...],
    time_splits: int,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    if time_splits <= 0:
        raise ValueError("time_splits must be positive.")
    if num_contents <= 0:
        raise ValueError("num_contents must be positive.")
    if num_users <= 0:
        raise ValueError("num_users must be positive.")

    if not events:
        return torch.zeros(1, dtype=torch.long), [_normalized_incidence(num_contents, num_users, [])]

    sorted_events = sorted(events, key=lambda item: item[3])
    num_segments = min(time_splits, len(sorted_events))
    cutoff_positions: list[int] = []
    for split_idx in range(1, num_segments + 1):
        cutoff_pos = (split_idx * len(sorted_events) + num_segments - 1) // num_segments - 1
        cutoff_positions.append(min(max(cutoff_pos, 0), len(sorted_events) - 1))

    segment_orders: list[int] = []
    segment_matrices: list[tuple[torch.Tensor, torch.Tensor]] = []
    start_pos = 0

    for cutoff_pos in cutoff_positions:
        cutoff_order = sorted_events[cutoff_pos][3]
        end_pos = cutoff_pos + 1
        while end_pos < len(sorted_events) and sorted_events[end_pos][3] == cutoff_order:
            end_pos += 1

        edges = [
            (content_idx, user_idx)
            for content_idx, user_idx, _, _ in sorted_events[start_pos:end_pos]
        ]
        segment_orders.append(cutoff_order)
        segment_matrices.append(_normalized_incidence(num_contents, num_users, edges))
        start_pos = end_pos

    return torch.tensor(segment_orders, dtype=torch.long), segment_matrices
