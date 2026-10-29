from __future__ import annotations

import csv
import hashlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_DATA_FILES = ("user.txt", "content.txt", "cascades.txt", "user_history.txt")
SPLIT_NAMES = ("train", "val", "test")
SAMPLE_POLICIES = ("all", "loo_train", "loo_val", "loo_test")


@dataclass(frozen=True)
class RawEvent:
    user_id: str
    timestamp: int
    action: str


@dataclass(frozen=True)
class RawCascade:
    content_id: str
    first_timestamp: int
    events: tuple[RawEvent, ...]


def _log_step(message: str, start_time: float | None = None) -> float:
    now = time.perf_counter()
    if start_time is None:
        print(f"[data] {message}", flush=True)
    else:
        print(f"[data] {message} ({now - start_time:.1f}s)", flush=True)
    return now


def _bucket_seconds(seconds: int, max_time_bucket: int) -> int:
    if seconds <= 0:
        return 0
    return min(int(math.log2(seconds + 1)), max_time_bucket)


def _parse_cascade_events(cascade_text: str) -> list[RawEvent]:
    events: list[RawEvent] = []
    for item in cascade_text.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = item.split()
        if len(parts) < 3:
            continue

        try:
            timestamp = int(parts[1])
        except ValueError:
            continue

        events.append(RawEvent(
            user_id=parts[0],
            timestamp=timestamp,
            action=" ".join(parts[2:]),
        ))

    if len(events) > 1:
        events.sort(key=lambda event: event.timestamp)
    return events


def _read_cascades(cascades_path: Path) -> list[RawCascade]:
    cascades: list[RawCascade] = []
    with cascades_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header[:2] != ["content_id", "cascade"]:
            raise ValueError(
                f"{cascades_path} must start with header: content_id<TAB>cascade"
            )

        for line_no, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                content_id, cascade_text = line.split("\t", 1)
            except ValueError:
                continue

            events = _parse_cascade_events(cascade_text)
            if len(events) < 2:
                continue

            cascades.append(RawCascade(
                content_id=content_id,
                first_timestamp=events[0].timestamp,
                events=tuple(events),
            ))

            if line_no % 100_000 == 0:
                print(f"[data] Parsed cascades: {len(cascades):,}", flush=True)

    cascades.sort(key=lambda cascade: (cascade.first_timestamp, cascade.content_id))
    return cascades


def _read_content_texts(content_path: Path) -> dict[str, str]:
    content_texts: dict[str, str] = {}
    text_fields = (
        "title",
        "text",
        "category",
        "tags",
        "content_type",
        "author_id",
        "extra",
    )
    with content_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "content_id" not in reader.fieldnames:
            raise ValueError(f"{content_path} must contain a content_id column.")
        for row in reader:
            content_id = row.get("content_id", "")
            if not content_id:
                continue
            parts = [
                (row.get(field) or "").strip()
                for field in text_fields
                if (row.get(field) or "").strip()
            ]
            content_texts[content_id] = " ".join(parts)
    return content_texts


def _safe_cache_name(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "model"


def _content_order_hash(content_ids: list[str]) -> str:
    digest = hashlib.sha1()
    for content_id in content_ids:
        digest.update(content_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _build_content_semantic_cache_path(
    data_path: Path,
    split_strategy: str,
    model_name: str,
    max_length: int,
    content_ids: list[str],
    cache_dir: str | None,
) -> Path:
    root = Path(cache_dir) if cache_dir else data_path / "cache"
    name = (
        f"content_t5_{_safe_cache_name(model_name)}"
        f"_len{max_length}_{split_strategy}"
        f"_n{len(content_ids)}_{_content_order_hash(content_ids)}.pt"
    )
    return root / name


def _mean_pool_encoder_output(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    pooled = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return pooled / denom


def _encode_content_texts_with_t5(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "Semantic content embeddings require the transformers package."
        ) from exc

    if batch_size <= 0:
        raise ValueError("content_semantic_batch_size must be positive.")
    if max_length <= 0:
        raise ValueError("content_semantic_max_length must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = T5EncoderModel.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        raise OSError(
            f"Could not load T5 semantic encoder {model_name!r} from the local Hugging Face cache. "
            "Set content_semantic_model to a cached model id or path, or pre-download "
            "the model before building semantic content embeddings."
        ) from exc
    model.to(device)
    model.eval()

    embeddings: list[torch.Tensor] = []
    total = len(texts)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = [text if text else "empty content" for text in texts[start:end]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = _mean_pool_encoder_output(output.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu())
            if end == total or end % max(batch_size * 20, 1024) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[data] Encoded content text with T5: {end:,}/{total:,} ({elapsed:.1f}s)",
                    flush=True,
                )

    return torch.cat(embeddings, dim=0).float()


def _load_or_build_content_semantic_embeddings(
    data_path: Path,
    split_strategy: str,
    train_cascades: list[RawCascade],
    model_name: str,
    batch_size: int,
    max_length: int,
    cache_dir: str | None,
) -> torch.Tensor:
    content_ids = [cascade.content_id for cascade in train_cascades]
    cache_path = _build_content_semantic_cache_path(
        data_path=data_path,
        split_strategy=split_strategy,
        model_name=model_name,
        max_length=max_length,
        content_ids=content_ids,
        cache_dir=cache_dir,
    )
    if cache_path.exists():
        print(f"[data] Loading cached T5 content embeddings: {cache_path}", flush=True)
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict):
            embeddings = cached.get("embeddings")
        else:
            embeddings = cached
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(f"Invalid content embedding cache: {cache_path}")
        if embeddings.shape[0] != len(content_ids):
            raise ValueError(
                f"Cached content embeddings row mismatch: expected {len(content_ids)}, "
                f"got {embeddings.shape[0]}"
            )
        return embeddings.float()

    content_texts = _read_content_texts(data_path / "content.txt")
    texts = [content_texts.get(content_id, "") for content_id in content_ids]
    missing = sum(1 for text in texts if not text)
    if missing:
        print(f"[data] Warning: missing semantic text for {missing:,} train contents.", flush=True)

    print(
        f"[data] Building T5 content embeddings: model={model_name}, "
        f"contents={len(texts):,}, max_length={max_length}",
        flush=True,
    )
    embeddings = _encode_content_texts_with_t5(
        texts=texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "max_length": max_length,
            "split_strategy": split_strategy,
            "content_ids": content_ids,
            "embeddings": embeddings,
        },
        cache_path,
    )
    print(f"[data] Saved T5 content embeddings: {cache_path}", flush=True)
    return embeddings


def _split_cascades(
    cascades: list[RawCascade],
    split_strategy: str,
) -> tuple[dict[str, list[RawCascade]], dict[str, str]]:
    if split_strategy == "temporal_content":
        n_cascades = len(cascades)
        train_end = int(n_cascades * 0.8)
        val_end = int(n_cascades * 0.9)
        return {
            "train": cascades[:train_end],
            "val": cascades[train_end:val_end],
            "test": cascades[val_end:],
        }, {
            "train": "all",
            "val": "all",
            "test": "all",
        }

    if split_strategy == "leave_one_out":
        shared = {
            "train": cascades,
            "val": cascades,
            "test": cascades,
        }
        return shared, {
            "train": "loo_train",
            "val": "loo_val",
            "test": "loo_test",
        }

    raise ValueError(
        f"Unsupported split_strategy={split_strategy!r}. "
        "Expected one of: temporal_content, leave_one_out"
    )


def _build_vocabs(train_cascades: list[RawCascade]) -> tuple[dict[str, int], dict[str, int]]:
    users: set[str] = set()
    actions: set[str] = set()
    for cascade in train_cascades:
        for event in cascade.events:
            users.add(event.user_id)
            actions.add(event.action)

    user2idx = {"__PAD__": 0, "__UNK__": 1}
    for idx, user_id in enumerate(sorted(users), start=2):
        user2idx[user_id] = idx

    action2idx = {"__PAD__": 0, "__UNK__": 1}
    for idx, action in enumerate(sorted(actions), start=2):
        action2idx[action] = idx

    return user2idx, action2idx


def _adopter_order_and_first_indices(cascade: RawCascade) -> tuple[list[str], list[int]]:
    seen_users: set[str] = set()
    adopter_order: list[str] = []
    adopter_first_event_idx: list[int] = []
    for event_idx, event in enumerate(cascade.events):
        if event.user_id in seen_users:
            continue
        seen_users.add(event.user_id)
        adopter_order.append(event.user_id)
        adopter_first_event_idx.append(event_idx)
    return adopter_order, adopter_first_event_idx


def _build_event_order_lookup(cascades: list[RawCascade]) -> dict[int, list[int]]:
    if not cascades:
        return {}

    timestamp_counts: dict[int, int] = {}
    for cascade in cascades:
        for event in cascade.events:
            timestamp_counts[event.timestamp] = timestamp_counts.get(event.timestamp, 0) + 1

    timestamp_offsets: dict[int, int] = {}
    next_order = 0
    for timestamp in sorted(timestamp_counts):
        timestamp_offsets[timestamp] = next_order
        next_order += timestamp_counts[timestamp]

    if next_order >= (1 << 63):
        raise ValueError("Too many cascade events to store event_order as int64.")

    lookup: dict[int, list[int]] = {}
    timestamp_seen: dict[int, int] = {}
    for cascade in cascades:
        cascade_orders: list[int] = []
        for event in cascade.events:
            offset = timestamp_offsets[event.timestamp]
            seen_count = timestamp_seen.get(event.timestamp, 0)
            cascade_orders.append(offset + seen_count)
            timestamp_seen[event.timestamp] = seen_count + 1
        lookup[id(cascade)] = cascade_orders
    return lookup


def _selected_adopter_indices(adopter_count: int, sample_policy: str):
    if sample_policy == "all":
        return range(1, adopter_count)
    if sample_policy == "loo_train":
        return range(1, max(adopter_count - 2, 1))
    if sample_policy == "loo_val":
        return [adopter_count - 2] if adopter_count >= 3 else []
    if sample_policy == "loo_test":
        return [adopter_count - 1] if adopter_count >= 2 else []
    raise ValueError(
        f"Unsupported sample_policy={sample_policy!r}. "
        f"Expected one of: {', '.join(SAMPLE_POLICIES)}"
    )


def _graph_visible_adopter_indices(adopter_count: int, sample_policy: str):
    if sample_policy == "all":
        return range(0, adopter_count)
    if sample_policy == "loo_train":
        return range(0, max(adopter_count - 2, 0))
    raise ValueError(
        f"Unsupported train graph sample_policy={sample_policy!r}. "
        "Expected one of: all, loo_train"
    )


def _build_train_content_user_edges(
    train_cascades: list[RawCascade],
    user2idx: dict[str, int],
    sample_policy: str = "all",
) -> tuple[int, list[tuple[int, int]]]:
    edges: list[tuple[int, int]] = []
    for content_idx, cascade in enumerate(train_cascades):
        seen_users: set[int] = set()
        adopter_order, _ = _adopter_order_and_first_indices(cascade)
        for adopter_idx in _graph_visible_adopter_indices(len(adopter_order), sample_policy):
            user_idx = user2idx.get(adopter_order[adopter_idx], 1)
            if user_idx < 2 or user_idx in seen_users:
                continue
            seen_users.add(user_idx)
            edges.append((content_idx, user_idx))
    return len(train_cascades), edges


def _build_train_content_user_events(
    train_cascades: list[RawCascade],
    user2idx: dict[str, int],
    event_order_lookup: dict[int, list[int]],
    sample_policy: str = "all",
) -> list[tuple[int, int, int, int]]:
    temporal_events: list[tuple[int, int, int, int]] = []
    for content_idx, cascade in enumerate(train_cascades):
        seen_users: set[int] = set()
        adopter_order, adopter_first_event_idx = _adopter_order_and_first_indices(cascade)
        event_orders = event_order_lookup[id(cascade)]
        for adopter_idx in _graph_visible_adopter_indices(len(adopter_order), sample_policy):
            user_idx = user2idx.get(adopter_order[adopter_idx], 1)
            if user_idx < 2 or user_idx in seen_users:
                continue
            seen_users.add(user_idx)
            event_idx = adopter_first_event_idx[adopter_idx]
            temporal_events.append((
                content_idx,
                user_idx,
                cascade.events[event_idx].timestamp,
                event_orders[event_idx],
            ))
    return temporal_events


class CascadePredictionDataset(Dataset):
    def __init__(
        self,
        cascades: list[RawCascade],
        user2idx: dict[str, int],
        action2idx: dict[str, int],
        graph_content_indices: list[int] | None = None,
        max_seq_len: int = 512,
        max_time_bucket: int = 32,
        shuffle_contents_seed: int | None = None,
        sample_policy: str = "all",
        include_timestamps: bool = False,
        event_order_lookup: dict[int, list[int]] | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.include_timestamps = include_timestamps
        self.events_by_content: list[dict[str, torch.Tensor]] = []
        self.sample_content_idx: list[int] = []
        self.sample_prefix_end: list[int] = []
        self.sample_target_user: list[int] = []
        self.n_events = 0
        self.cold_start_samples = 0
        self.sample_policy = sample_policy

        if sample_policy not in SAMPLE_POLICIES:
            raise ValueError(
                f"Unsupported sample_policy={sample_policy!r}. "
                f"Expected one of: {', '.join(SAMPLE_POLICIES)}"
            )

        if graph_content_indices is None:
            graph_content_indices = [-1] * len(cascades)
        if len(graph_content_indices) != len(cascades):
            raise ValueError("graph_content_indices must align with cascades.")
        if include_timestamps and event_order_lookup is None:
            raise ValueError("event_order_lookup is required when include_timestamps=True.")

        ordered_pairs = list(zip(cascades, graph_content_indices))
        if shuffle_contents_seed is not None:
            random.Random(shuffle_contents_seed).shuffle(ordered_pairs)

        for cascade, graph_content_idx in ordered_pairs:
            event_orders = event_order_lookup[id(cascade)] if event_order_lookup is not None else None
            self._add_cascade(
                cascade,
                graph_content_idx,
                user2idx,
                action2idx,
                max_time_bucket,
                event_orders,
            )

    def _add_cascade(
        self,
        cascade: RawCascade,
        graph_content_idx: int,
        user2idx: dict[str, int],
        action2idx: dict[str, int],
        max_time_bucket: int,
        event_orders: list[int] | None,
    ) -> None:
        events = cascade.events
        if len(events) < 2:
            return

        first_timestamp = events[0].timestamp
        previous_timestamp = first_timestamp

        user_ids: list[int] = []
        action_ids: list[int] = []
        event_age_buckets: list[int] = []
        inter_delta_buckets: list[int] = []
        timestamps: list[int] = []
        event_order_values: list[int] = []

        for event_idx, event in enumerate(events):
            if event_idx == 0:
                delta_seconds = 0
            else:
                delta_seconds = max(event.timestamp - previous_timestamp, 0)

            user_ids.append(user2idx.get(event.user_id, 1))
            action_ids.append(action2idx.get(event.action, 1))
            event_age_buckets.append(_bucket_seconds(event.timestamp - first_timestamp, max_time_bucket))
            inter_delta_buckets.append(_bucket_seconds(delta_seconds, max_time_bucket))
            if self.include_timestamps:
                timestamps.append(event.timestamp)
                if event_orders is None:
                    raise ValueError("event_orders must be provided when timestamps are included.")
                event_order_values.append(event_orders[event_idx])
            previous_timestamp = event.timestamp

        content_idx = len(self.events_by_content)
        content_events = {
            "user_id": torch.tensor(user_ids, dtype=torch.long),
            "action_id": torch.tensor(action_ids, dtype=torch.long),
            "event_age_bucket": torch.tensor(event_age_buckets, dtype=torch.long),
            "inter_delta_bucket": torch.tensor(inter_delta_buckets, dtype=torch.long),
            "content_idx": torch.tensor(graph_content_idx, dtype=torch.long),
        }
        if self.include_timestamps:
            content_events["timestamp"] = torch.tensor(timestamps, dtype=torch.long)
            content_events["event_order"] = torch.tensor(event_order_values, dtype=torch.long)
        self.events_by_content.append(content_events)
        self.n_events += len(events)

        adopter_order, adopter_first_event_idx = _adopter_order_and_first_indices(cascade)
        selected_adopter_idxs = _selected_adopter_indices(len(adopter_order), self.sample_policy)

        for adopter_idx in selected_adopter_idxs:
            prefix_end = adopter_first_event_idx[adopter_idx] - 1
            if prefix_end < 0:
                continue

            target_user = user2idx.get(adopter_order[adopter_idx], 1)
            if target_user == 1:
                self.cold_start_samples += 1

            self.sample_content_idx.append(content_idx)
            self.sample_prefix_end.append(prefix_end)
            self.sample_target_user.append(target_user)

    def __len__(self) -> int:
        return len(self.sample_content_idx)

    def __getitem__(self, idx: int) -> dict:
        content_idx = self.sample_content_idx[idx]
        prefix_end = self.sample_prefix_end[idx]
        target_user = self.sample_target_user[idx]

        events = self.events_by_content[content_idx]
        prefix_stop = prefix_end + 1
        prefix_start = max(prefix_stop - self.max_seq_len, 0)

        item = {
            "prefix_users": events["user_id"][prefix_start:prefix_stop],
            "prefix_actions": events["action_id"][prefix_start:prefix_stop],
            "event_age_buckets": events["event_age_bucket"][prefix_start:prefix_stop],
            "inter_delta_buckets": events["inter_delta_bucket"][prefix_start:prefix_stop],
            "content_idx": events["content_idx"],
            "target_user": target_user,
        }
        if self.include_timestamps:
            item["prefix_timestamps"] = events["timestamp"][prefix_start:prefix_stop]
            item["prefix_event_orders"] = events["event_order"][prefix_start:prefix_stop]
        return item

    def stats(self) -> dict[str, int | float]:
        n_samples = len(self)
        return {
            "n_contents": len(self.events_by_content),
            "n_events": self.n_events,
            "n_samples": n_samples,
            "cold_start_samples": self.cold_start_samples,
            "cold_start_ratio": self.cold_start_samples / n_samples if n_samples else 0.0,
        }


class CascadeBatchCollator:
    def __init__(self, max_seq_len: int = 512) -> None:
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[dict]) -> dict:
        return _collate_cascade_batch(batch, self.max_seq_len)


def _collate_cascade_batch(batch: list[dict], max_seq_len: int = 512) -> dict:
    batch_size = len(batch)
    max_len = max((b["prefix_users"].shape[0] for b in batch), default=0)
    max_len = min(max_len, max_seq_len)

    prefix_users = torch.zeros(batch_size, max_len, dtype=torch.long)
    prefix_actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    event_age_buckets = torch.zeros(batch_size, max_len, dtype=torch.long)
    inter_delta_buckets = torch.zeros(batch_size, max_len, dtype=torch.long)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    target_users = torch.zeros(batch_size, dtype=torch.long)
    content_idx = torch.full((batch_size,), -1, dtype=torch.long)
    include_timestamps = bool(batch and "prefix_timestamps" in batch[0])
    prefix_timestamps = torch.full((batch_size, max_len), -1, dtype=torch.long) if include_timestamps else None
    prefix_event_orders = torch.full((batch_size, max_len), -1, dtype=torch.long) if include_timestamps else None

    for row_idx, item in enumerate(batch):
        target_users[row_idx] = item["target_user"]
        content_idx[row_idx] = item["content_idx"]
        length = min(item["prefix_users"].shape[0], max_seq_len)
        if length == 0:
            continue
        prefix_users[row_idx, :length] = item["prefix_users"][:length]
        prefix_actions[row_idx, :length] = item["prefix_actions"][:length]
        event_age_buckets[row_idx, :length] = item["event_age_buckets"][:length]
        inter_delta_buckets[row_idx, :length] = item["inter_delta_buckets"][:length]
        if prefix_timestamps is not None:
            prefix_timestamps[row_idx, :length] = item["prefix_timestamps"][:length]
        if prefix_event_orders is not None:
            prefix_event_orders[row_idx, :length] = item["prefix_event_orders"][:length]
        key_padding_mask[row_idx, :length] = False

    batch_dict = {
        "prefix_users": prefix_users,
        "prefix_actions": prefix_actions,
        "event_age_buckets": event_age_buckets,
        "inter_delta_buckets": inter_delta_buckets,
        "key_padding_mask": key_padding_mask,
        "content_idx": content_idx,
        "target_user": target_users,
    }
    if prefix_timestamps is not None:
        batch_dict["prefix_timestamps"] = prefix_timestamps
    if prefix_event_orders is not None:
        batch_dict["prefix_event_orders"] = prefix_event_orders
    return batch_dict


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _make_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    meta: dict,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    max_seq_len: int,
    seed: int,
) -> tuple:
    collate = CascadeBatchCollator(max_seq_len)
    pin_memory = torch.cuda.is_available()
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 2,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed),
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed + 1),
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=_make_generator(seed + 2),
        **worker_kwargs,
    )
    return train_loader, val_loader, test_loader, meta


def _build_cascade_datasets(
    data_path: Path,
    max_seq_len: int = 512,
    seed: int = 42,
    max_time_bucket: int = 32,
    split_strategy: str = "temporal_content",
    build_content_user_graph: bool = False,
    build_temporal_content_user_graph: bool = False,
    build_content_semantic_embeddings: bool = False,
    content_semantic_model: str = "sentence-transformers/sentence-t5-xxl",
    content_semantic_batch_size: int = 4,
    content_semantic_max_length: int = 128,
    content_semantic_cache_dir: str | None = None,
) -> tuple[dict[str, CascadePredictionDataset], dict]:
    missing = [name for name in REQUIRED_DATA_FILES if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{data_path} is missing required cascade data files: {', '.join(missing)}"
        )

    total_start = _log_step(f"Loading cascade prediction dataset: {data_path}")
    step_start = _log_step("Reading cascade events")
    cascades = _read_cascades(data_path / "cascades.txt")
    step_start = _log_step(f"Read valid cascades: rows={len(cascades):,}", step_start)
    if not cascades:
        raise ValueError(f"{data_path / 'cascades.txt'} contains no valid cascades")

    split_cascades, sample_policies = _split_cascades(cascades, split_strategy)
    step_start = _log_step(
        f"Computed {split_strategy} splits: "
        f"train={len(split_cascades['train']):,}, "
        f"val={len(split_cascades['val']):,}, "
        f"test={len(split_cascades['test']):,}",
        step_start,
    )

    _log_step("Building train-only user/action vocab")
    user2idx, action2idx = _build_vocabs(split_cascades["train"])
    step_start = _log_step(
        f"Built vocab: users={len(user2idx):,}, actions={len(action2idx):,}",
        step_start,
    )

    event_order_lookup = None
    if build_temporal_content_user_graph:
        order_start = _log_step("Indexing cascade event order")
        event_order_lookup = _build_event_order_lookup(cascades)
        _log_step("Indexed cascade event order", order_start)

    datasets: dict[str, CascadePredictionDataset] = {}
    stats: dict[str, dict[str, int | float]] = {}
    train_content_graph_idx = {id(cascade): idx for idx, cascade in enumerate(split_cascades["train"])}
    for split_offset, split_name in enumerate(SPLIT_NAMES):
        split_start = _log_step(f"Building {split_name} next-adopter samples")
        graph_content_indices = [
            train_content_graph_idx.get(id(cascade), -1)
            for cascade in split_cascades[split_name]
        ]
        split_ds = CascadePredictionDataset(
            split_cascades[split_name],
            user2idx=user2idx,
            action2idx=action2idx,
            graph_content_indices=graph_content_indices,
            max_seq_len=max_seq_len,
            max_time_bucket=max_time_bucket,
            shuffle_contents_seed=seed + split_offset,
            sample_policy=sample_policies[split_name],
            include_timestamps=build_temporal_content_user_graph,
            event_order_lookup=event_order_lookup,
        )
        datasets[split_name] = split_ds
        stats[split_name] = split_ds.stats()
        _log_step(
            f"Built {split_name}: "
            f"contents={stats[split_name]['n_contents']:,}, "
            f"events={stats[split_name]['n_events']:,}, "
            f"samples={stats[split_name]['n_samples']:,}",
            split_start,
        )

    meta = {
        "dataset": data_path.name,
        "seed": seed,
        "num_users": len(user2idx),
        "num_actions": len(action2idx),
        "max_time_bucket": max_time_bucket,
        "split_strategy": split_strategy,
        "source": "cascade_prediction",
        "files": {name.removesuffix(".txt"): str(data_path / name) for name in REQUIRED_DATA_FILES},
        "splits": stats,
    }
    if build_temporal_content_user_graph:
        graph_start = _log_step("Building train temporal content-user graph events")
        temporal_events = _build_train_content_user_events(
            split_cascades["train"],
            user2idx,
            event_order_lookup,
            sample_policy=sample_policies["train"],
        )
        num_contents = len(split_cascades["train"])
        meta["num_contents"] = num_contents
        meta["train_content_user_events"] = temporal_events
        _log_step(
            f"Prepared train temporal graph events: contents={num_contents:,}, events={len(temporal_events):,}",
            graph_start,
        )
        if build_content_user_graph:
            edges = [(content_idx, user_idx) for content_idx, user_idx, _, _ in temporal_events]
            meta["train_content_user_edges"] = edges
            _log_step(
                f"Built train graph edges: contents={num_contents:,}, "
                f"edges={len(edges):,}",
            )
    elif build_content_user_graph:
        graph_start = _log_step("Building train content-user graph edges")
        num_contents, edges = _build_train_content_user_edges(
            split_cascades["train"],
            user2idx,
            sample_policy=sample_policies["train"],
        )
        meta["num_contents"] = num_contents
        meta["train_content_user_edges"] = edges
        _log_step(
            f"Built train graph edges: contents={num_contents:,}, "
            f"edges={len(edges):,}",
            graph_start,
        )

    if build_content_semantic_embeddings:
        semantic_start = _log_step("Preparing train content semantic embeddings")
        content_semantic_embeddings = _load_or_build_content_semantic_embeddings(
            data_path=data_path,
            split_strategy=split_strategy,
            train_cascades=split_cascades["train"],
            model_name=content_semantic_model,
            batch_size=content_semantic_batch_size,
            max_length=content_semantic_max_length,
            cache_dir=content_semantic_cache_dir,
        )
        meta["content_semantic_embeddings"] = content_semantic_embeddings
        meta["content_semantic_model"] = content_semantic_model
        _log_step(
            f"Prepared semantic embeddings: shape={tuple(content_semantic_embeddings.shape)}",
            semantic_start,
        )
    _log_step("Cascade dataloader preparation done", total_start)
    return datasets, meta


def create_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    max_seq_len: int = 512,
    seed: int = 42,
    max_time_bucket: int = 32,
    split_strategy: str = "leave_one_out",
    build_content_user_graph: bool = False,
    build_temporal_content_user_graph: bool = False,
    build_content_semantic_embeddings: bool = False,
    content_semantic_model: str = "sentence-transformers/sentence-t5-xxl",
    content_semantic_batch_size: int = 4,
    content_semantic_max_length: int = 128,
    content_semantic_cache_dir: str | None = None,
) -> tuple:
    data_path = Path(data_dir)
    datasets, meta = _build_cascade_datasets(
        data_path=data_path,
        max_seq_len=max_seq_len,
        seed=seed,
        max_time_bucket=max_time_bucket,
        split_strategy=split_strategy,
        build_content_user_graph=build_content_user_graph,
        build_temporal_content_user_graph=build_temporal_content_user_graph,
        build_content_semantic_embeddings=build_content_semantic_embeddings,
        content_semantic_model=content_semantic_model,
        content_semantic_batch_size=content_semantic_batch_size,
        content_semantic_max_length=content_semantic_max_length,
        content_semantic_cache_dir=content_semantic_cache_dir,
    )

    if eval_batch_size is None:
        eval_batch_size = batch_size

    return _make_loaders(
        train_ds=datasets["train"],
        val_ds=datasets["val"],
        test_ds=datasets["test"],
        meta=meta,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_seq_len=max_seq_len,
        seed=seed,
    )
