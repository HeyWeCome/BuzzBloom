from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


MODEL_REGISTRY: dict[str, type] = {}
_MODELS_DISCOVERED = False


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def requires_content_user_graph(cls):
    cls.requires_content_user_graph = True
    return cls


def requires_temporal_content_user_graph(cls):
    cls.requires_temporal_content_user_graph = True
    return cls


def requires_content_semantic_embeddings(cls):
    cls.requires_content_semantic_embeddings = True
    return cls


def discover_models() -> None:
    global _MODELS_DISCOVERED
    if _MODELS_DISCOVERED:
        return

    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        if path.stem == "__init__":
            continue
        importlib.import_module(f"{__name__}.{path.stem}")
    _MODELS_DISCOVERED = True


def model_names() -> list[str]:
    discover_models()
    return sorted(MODEL_REGISTRY)


def get_model_class(name: str) -> type:
    discover_models()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {model_names()}")
    return MODEL_REGISTRY[name]


def model_requires_content_user_graph(name: str) -> bool:
    cls = get_model_class(name)
    return bool(getattr(cls, "requires_content_user_graph", False))


def model_requires_temporal_content_user_graph(name: str) -> bool:
    cls = get_model_class(name)
    return bool(getattr(cls, "requires_temporal_content_user_graph", False))


def model_requires_content_semantic_embeddings(name: str) -> bool:
    cls = get_model_class(name)
    return bool(getattr(cls, "requires_content_semantic_embeddings", False))


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Model")
    group.add_argument("--d_model", type=int, default=64)
    group.add_argument("--num_layers", type=int, default=2)
    group.add_argument("--num_heads", type=int, default=8)
    group.add_argument("--dropout", type=float, default=0.1)
    group.add_argument("--max_seq_len", type=int, default=32)
    group.add_argument("--max_time_bucket", type=int, default=32)
    group.add_argument("--normalize_embeddings", action="store_true")
    group.add_argument("--logit_scale", type=float, default=1.0)
    group.add_argument("--learnable_scale", action="store_true")


def add_model_args(parser: argparse.ArgumentParser, name: str) -> None:
    add_args = getattr(get_model_class(name), "add_model_args", None)
    if add_args is not None:
        add_args(parser)


def build_common_model_kwargs(args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_users": meta["num_users"],
        "num_actions": meta["num_actions"],
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "max_time_bucket": args.max_time_bucket,
        "normalize_embeddings": args.normalize_embeddings,
        "logit_scale": args.logit_scale,
        "learnable_scale": args.learnable_scale,
    }


def build_model_kwargs(name: str, args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    cls = get_model_class(name)
    kwargs = build_common_model_kwargs(args, meta)
    if model_requires_content_user_graph(name):
        kwargs.update({
            "num_contents": meta["num_contents"],
            "train_content_user_edges": meta["train_content_user_edges"],
        })
    if model_requires_temporal_content_user_graph(name):
        kwargs["num_contents"] = meta["num_contents"]
        kwargs["train_content_user_events"] = meta["train_content_user_events"]
    if model_requires_content_semantic_embeddings(name):
        kwargs["content_semantic_embeddings"] = meta["content_semantic_embeddings"]

    extend_kwargs = getattr(cls, "build_model_kwargs", None)
    if extend_kwargs is not None:
        kwargs.update(extend_kwargs(args, meta))
    return kwargs


def build_model_from_args(name: str, args: argparse.Namespace, meta: dict[str, Any]):
    return get_model_class(name)(**build_model_kwargs(name, args, meta))


def model_run_name_parts(name: str, args: argparse.Namespace) -> list[Any]:
    build_parts = getattr(get_model_class(name), "run_name_parts", None)
    if build_parts is None:
        return []
    return list(build_parts(args))


def build_model(name: str, **kwargs):
    return get_model_class(name)(**kwargs)
