from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch

from dataset.dataloader import create_dataloaders
from models import (
    add_common_model_args,
    add_model_args,
    build_model_from_args,
    model_names,
    model_requires_content_semantic_embeddings,
    model_requires_content_user_graph,
    model_requires_temporal_content_user_graph,
    model_run_name_parts,
)
from trainer import CascadeLightningModule, create_trainer
from utils import build_run_dir, save_config, set_seed


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_num_workers(num_workers: int, device: str) -> int:
    if num_workers >= 0:
        return num_workers
    cpu_count = os.cpu_count() or 1
    if device == "cuda":
        return min(8, max(2, cpu_count // 2))
    return max(1, min(4, cpu_count // 2))


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--model", default="alter")
    bootstrap_args, _ = bootstrap.parse_known_args()
    available_models = model_names()
    if bootstrap_args.model not in available_models:
        bootstrap.error(
            f"argument --model: invalid choice: {bootstrap_args.model!r} "
            f"(choose from {', '.join(available_models)})"
        )

    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train", choices=["train", "eval"])
    p.add_argument("--model", default=bootstrap_args.model, choices=available_models)
    p.add_argument("--data", default="kuairand")
    p.add_argument("--data_root", "--data-root", default="./data")
    p.add_argument(
        "--split_strategy",
        default="leave_one_out",
        choices=["temporal_content", "leave_one_out"],
    )
    add_common_model_args(p)
    add_model_args(p, bootstrap_args.model)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_negatives", type=int, default=100)
    p.add_argument("--eval_k", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--test_after_fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run test with the best checkpoint after training. Disable during hyperparameter search.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=-1)
    p.add_argument("--device", default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium"])
    p.add_argument("--log_dir", default="./runs")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = _detect_device() if args.device == "auto" else args.device
    args.device = device
    args.num_workers = _resolve_num_workers(args.num_workers, device)
    if device == "cuda":
        torch.set_float32_matmul_precision(args.matmul_precision)
    data_dir = Path(args.data_root) / args.data
    build_content_user_graph = model_requires_content_user_graph(args.model)
    build_temporal_content_user_graph = model_requires_temporal_content_user_graph(args.model)
    build_content_semantic_embeddings = model_requires_content_semantic_embeddings(args.model)

    print(f"[main] Using device: {device}", flush=True)
    print(f"[main] Trainer devices: {args.devices}", flush=True)
    print(f"[main] Deterministic training: {args.deterministic}", flush=True)
    print(f"[main] DataLoader num_workers: {args.num_workers}", flush=True)
    if args.eval_batch_size <= 0:
        args.eval_batch_size = min(args.batch_size, 16) if build_content_user_graph else args.batch_size
    print(f"[main] Eval batch_size: {args.eval_batch_size}", flush=True)

    print(f"[main] Dataset: {args.data} ({data_dir})", flush=True)

    train_loader, val_loader, test_loader, meta = create_dataloaders(
        str(data_dir),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        max_time_bucket=args.max_time_bucket,
        split_strategy=args.split_strategy,
        build_content_user_graph=build_content_user_graph,
        build_temporal_content_user_graph=build_temporal_content_user_graph,
        build_content_semantic_embeddings=build_content_semantic_embeddings,
        content_semantic_model=getattr(
            args,
            "content_semantic_model",
            "sentence-transformers/sentence-t5-xxl",
        ),
        content_semantic_batch_size=getattr(
            args,
            "content_semantic_batch_size",
            4,
        ),
        content_semantic_max_length=getattr(
            args,
            "content_semantic_max_length",
            128,
        ),
        content_semantic_cache_dir=getattr(
            args,
            "content_semantic_cache_dir",
            None,
        ),
    )

    model = build_model_from_args(args.model, args, meta)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"[main] Parameters: trainable={trainable_parameters:,}, total={total_parameters:,}",
        flush=True,
    )

    lit_module = CascadeLightningModule(
        model,
        num_negatives=args.num_negatives,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_users=meta["num_users"],
        eval_k=args.eval_k,
    )

    log_dir = str(build_run_dir(args, model_run_name_parts(args.model, args)))
    print(f"[main] Run directory: {log_dir}", flush=True)
    save_config(args, log_dir)
    trainer = create_trainer(args, log_dir)

    if args.mode == "train":
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()
        trainer.fit(lit_module, train_loader, val_loader)
        train_seconds = time.perf_counter() - train_start
        print(f"[main] Training wall time: {train_seconds:.1f}s", flush=True)
        if device == "cuda":
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[main] Peak CUDA memory allocated: {peak_gib:.3f} GiB", flush=True)
        if args.test_after_fit:
            trainer.test(lit_module, test_loader, ckpt_path="best")
        else:
            print("[main] Skipping test after fit.", flush=True)
    else:
        trainer.test(lit_module, test_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
