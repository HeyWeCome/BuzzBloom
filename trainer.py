from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def _build_train_user_tensor(num_users: int, device: torch.device) -> torch.Tensor:
    return torch.arange(2, num_users, device=device, dtype=torch.long)


def _sample_negatives_strict(
    train_users: torch.Tensor,
    target_users: torch.Tensor,
    prefix_users: torch.Tensor,
    key_padding_mask: torch.Tensor,
    num_negatives: int,
) -> torch.Tensor:
    B = target_users.shape[0]
    device = target_users.device
    pool_size = train_users.shape[0]
    weights = torch.ones(B, pool_size, dtype=torch.float32, device=device)

    target_idx = target_users - 2
    target_valid = (target_idx >= 0) & (target_idx < pool_size)
    weights[torch.arange(B, device=device)[target_valid], target_idx[target_valid]] = 0.0

    if prefix_users is not None and key_padding_mask is not None and prefix_users.numel() > 0:
        prefix_idx = prefix_users - 2
        prefix_valid = (~key_padding_mask) & (prefix_idx >= 0) & (prefix_idx < pool_size)
        rows = torch.arange(B, device=device).unsqueeze(1).expand_as(prefix_idx)
        weights[rows[prefix_valid], prefix_idx[prefix_valid]] = 0.0

    legal_counts = (weights > 0).sum(dim=1)
    if torch.any(legal_counts == 0):
        raise RuntimeError("No legal negative users available for a sample.")

    if torch.any(legal_counts < num_negatives):
        sampled = torch.multinomial(weights, num_samples=num_negatives, replacement=True)
        return train_users[sampled]

    sampled = torch.multinomial(weights, num_samples=num_negatives, replacement=False)
    return train_users[sampled]


def _mask_seen_prefix_users(
    scores: torch.Tensor,
    prefix_users: torch.Tensor,
    key_padding_mask: torch.Tensor,
    target_users: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if prefix_users.numel() == 0:
        return scores

    prefix_idx = prefix_users - 2
    prefix_valid = (~key_padding_mask) & (prefix_idx >= 0) & (prefix_idx < scores.shape[1])
    if prefix_valid.any():
        rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(prefix_idx)
        scores[rows[prefix_valid], prefix_idx[prefix_valid]] = float("-inf")

    if target_users is not None:
        target_idx = target_users - 2
        target_valid = (target_idx >= 0) & (target_idx < scores.shape[1])
        if torch.any(target_valid):
            scores[torch.arange(scores.shape[0], device=scores.device)[target_valid], target_idx[target_valid]] = (
                scores[torch.arange(scores.shape[0], device=scores.device)[target_valid], target_idx[target_valid]]
            )

    return scores


class CascadeLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_negatives: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        num_users: int = 0,
        eval_k: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_negatives = num_negatives
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_users = num_users
        self.eval_k = eval_k

        self._train_users: Optional[torch.Tensor] = None

        self.register_buffer("_val_hit_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_ndcg_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_mrr_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_val_count", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_hit_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_hit20_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_recall_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_ndcg_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_ndcg20_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_mrr_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_test_count", torch.tensor(0.0), persistent=False)

    def _get_train_users(self, device: torch.device) -> torch.Tensor:
        if self._train_users is None or self._train_users.device != device:
            self._train_users = _build_train_user_tensor(self.num_users, device)
        return self._train_users

    def on_train_epoch_start(self) -> None:
        if hasattr(self.model, "before_epoch"):
            self.model.before_epoch()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        prefix_users = batch["prefix_users"]
        prefix_actions = batch["prefix_actions"]
        event_age = batch["event_age_buckets"]
        inter_delta = batch["inter_delta_buckets"]
        key_padding_mask = batch["key_padding_mask"]
        target_user = batch["target_user"]

        if hasattr(self.model, "compute_loss"):
            loss = self.model.compute_loss(batch, num_negatives=self.num_negatives)
        elif hasattr(self.model, "score_all_train_users"):
            scores = self.model.score_all_train_users(
                prefix_users, prefix_actions, event_age, inter_delta, key_padding_mask,
            )
            scores = _mask_seen_prefix_users(scores, prefix_users, key_padding_mask, target_user)
            loss = F.cross_entropy(scores, target_user - 2)
        else:
            train_users = self._get_train_users(prefix_users.device)
            negatives = _sample_negatives_strict(
                train_users, target_user, prefix_users, key_padding_mask, self.num_negatives,
            )
            candidate_users = torch.cat([target_user.unsqueeze(-1), negatives], dim=1)

            scores = self.model(
                prefix_users, prefix_actions, event_age, inter_delta,
                candidate_users, key_padding_mask,
            )
            loss = F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch: dict, stage: str) -> None:
        prefix_users = batch["prefix_users"]
        prefix_actions = batch["prefix_actions"]
        event_age = batch["event_age_buckets"]
        inter_delta = batch["inter_delta_buckets"]
        key_padding_mask = batch["key_padding_mask"]
        target_user = batch["target_user"]

        B = target_user.shape[0]
        if hasattr(self.model, "score_all_train_users_from_batch"):
            scores = self.model.score_all_train_users_from_batch(batch)
        elif hasattr(self.model, "score_all_train_users"):
            scores = self.model.score_all_train_users(
                prefix_users, prefix_actions, event_age, inter_delta, key_padding_mask,
            )
        else:
            raise RuntimeError("Full-user eval requires a model-level full ranking method.")
        scores = _mask_seen_prefix_users(scores, prefix_users, key_padding_mask, target_user)

        cold_ratio = (target_user == 1).float().mean()
        self.log(f"{stage}/cold_start_ratio", cold_ratio, on_epoch=True, batch_size=B)

        target_idx = target_user - 2
        seen_mask = (target_idx >= 0) & (target_idx < scores.shape[1])
        seen_count = seen_mask.sum().item()
        if seen_count == 0:
            return

        seen_scores = scores[seen_mask]
        seen_target_idx = target_idx[seen_mask]
        target_scores = seen_scores.gather(1, seen_target_idx.unsqueeze(1))
        ranks = (seen_scores > target_scores).sum(dim=1) + 1
        ranks_f = ranks.float()

        hit_sum = (ranks <= 10).float().sum()
        hit20_sum = (ranks <= 20).float().sum()
        ndcg_sum = torch.where(
            ranks <= 10,
            1.0 / torch.log2(ranks_f + 1.0),
            torch.zeros_like(ranks_f),
        ).sum()
        ndcg20_sum = torch.where(
            ranks <= 20,
            1.0 / torch.log2(ranks_f + 1.0),
            torch.zeros_like(ranks_f),
        ).sum()
        mrr_sum = (1.0 / ranks_f).sum()

        getattr(self, f"_{stage}_hit_sum").add_(hit_sum)
        hit20_buffer = getattr(self, f"_{stage}_hit20_sum", None)
        if hit20_buffer is not None:
            hit20_buffer.add_(hit20_sum)
        recall_buffer = getattr(self, f"_{stage}_recall_sum", None)
        if recall_buffer is not None:
            recall_buffer.add_(hit_sum)
        getattr(self, f"_{stage}_ndcg_sum").add_(ndcg_sum)
        ndcg20_buffer = getattr(self, f"_{stage}_ndcg20_sum", None)
        if ndcg20_buffer is not None:
            ndcg20_buffer.add_(ndcg20_sum)
        getattr(self, f"_{stage}_mrr_sum").add_(mrr_sum)
        getattr(self, f"_{stage}_count").add_(float(seen_count))

    def _log_eval_epoch(self, stage: str, prog_bar: bool = False) -> None:
        hit_sum = getattr(self, f"_{stage}_hit_sum")
        hit20_sum = getattr(self, f"_{stage}_hit20_sum", None)
        recall_sum = getattr(self, f"_{stage}_recall_sum", None)
        ndcg_sum = getattr(self, f"_{stage}_ndcg_sum")
        ndcg20_sum = getattr(self, f"_{stage}_ndcg20_sum", None)
        mrr_sum = getattr(self, f"_{stage}_mrr_sum")
        count = getattr(self, f"_{stage}_count").clamp(min=1.0)

        self.log(f"{stage}/hit@10", hit_sum / count, on_epoch=True, prog_bar=prog_bar, sync_dist=True)
        if hit20_sum is not None:
            self.log(f"{stage}/hit@20", hit20_sum / count, on_epoch=True, sync_dist=True)
        if recall_sum is not None:
            self.log(f"{stage}/recall@10", recall_sum / count, on_epoch=True, prog_bar=prog_bar, sync_dist=True)
        self.log(f"{stage}/ndcg@10", ndcg_sum / count, on_epoch=True, prog_bar=prog_bar, sync_dist=True)
        if ndcg20_sum is not None:
            self.log(f"{stage}/ndcg@20", ndcg20_sum / count, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/mrr", mrr_sum / count, on_epoch=True, sync_dist=True)

        hit_sum.zero_()
        if hit20_sum is not None:
            hit20_sum.zero_()
        if recall_sum is not None:
            recall_sum.zero_()
        ndcg_sum.zero_()
        if ndcg20_sum is not None:
            ndcg20_sum.zero_()
        mrr_sum.zero_()
        getattr(self, f"_{stage}_count").zero_()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._log_eval_epoch("val", prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._log_eval_epoch("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 50,
        )
        return [optimizer], [scheduler]


def create_trainer(args, log_dir: str) -> pl.Trainer:
    callbacks: list[pl.callbacks.Callback] = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/ndcg@10", mode="max", save_top_k=1,
            dirpath=f"{log_dir}/checkpoints", filename="best",
        ),
        pl.callbacks.ModelCheckpoint(
            save_last=True,
            dirpath=f"{log_dir}/checkpoints", filename="last-{epoch:02d}",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/ndcg@10", mode="max", patience=args.patience,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    return pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=args.devices if args.device == "cuda" else 1,
        precision="16-mixed" if args.amp else "32-true",
        deterministic=args.deterministic,
        benchmark=False,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=log_dir, name="metrics"),
        log_every_n_steps=args.log_interval,
        enable_model_summary=not args.amp,
    )
