from __future__ import annotations

import os
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def save_config(args: Namespace, log_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in sorted(vars(args).items()):
        lines.append(f"{key}: {value}")
    (log_dir / "config.yaml").write_text("\n".join(lines), encoding="utf-8")


def format_run_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = f"{value:g}"
        text = text.replace("e-0", "e-").replace("e-", "e-")
        text = text.replace("e+0", "e").replace("e+", "e")
        return text.replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def slugify_run_part(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "na"


def format_run_part(part: Any) -> str:
    if isinstance(part, tuple) and len(part) == 2:
        key, value = part
        return f"{slugify_run_part(key)}{format_run_value(value)}"
    return slugify_run_part(part)


def build_run_name(args: Namespace, model_parts: Iterable[Any] = ()) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        timestamp,
        slugify_run_part(args.model),
        slugify_run_part(args.data),
        f"split-{slugify_run_part(args.split_strategy)}",
        f"d{args.d_model}",
        f"l{args.num_layers}",
        f"h{args.num_heads}",
        f"seq{args.max_seq_len}",
        f"bs{args.batch_size}",
        f"ebs{args.eval_batch_size}",
        f"lr{format_run_value(args.lr)}",
        f"wd{format_run_value(args.weight_decay)}",
        f"neg{args.num_negatives}",
        f"seed{args.seed}",
    ]
    parts.extend(format_run_part(part) for part in model_parts)
    return "_".join(parts)


def build_run_dir(args: Namespace, model_parts: Iterable[Any] = ()) -> Path:
    root = Path(args.log_dir)
    run_name = build_run_name(args, model_parts)
    run_dir = root / run_name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{run_name}_{suffix:02d}"
        suffix += 1
    return run_dir
