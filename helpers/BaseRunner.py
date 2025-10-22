import logging
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# Removed AMP imports:
# from torch.amp import autocast, GradScaler

from utils import Constants
from utils.Metrics import Metrics
from utils.Optim import ScheduledOptim


class BaseRunner(object):
    """
    Handles the training, validation, and testing pipeline for a model.
    This version uses standard full-precision (float32) training.
    """

    def __init__(self, args):
        # Stop training if the validation score doesn't improve for `patience` epochs.
        self.patience = args.patience
        # The 'use_amp' flag has been removed as this runner no longer uses mixed precision.

    def run(self, model, train_data, valid_data, test_data, data_loader, args):
        """
        Executes the main training and evaluation loop.
        """
        loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=Constants.PAD)
        optimizer = ScheduledOptim(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            args.d_model,
            args.n_warmup_steps
        )

        model.to(args.device)
        loss_func.to(args.device)

        validation_history = 0.0
        best_scores = {}
        epochs_without_improvement = 0

        for epoch_i in range(args.epoch):
            if hasattr(model, 'before_epoch'):
                model.before_epoch()

            logging.info(f'\n[ Epoch {epoch_i} ]')
            start = time.time()

            # Train the model for one epoch
            train_loss, train_accu = self.train_epoch(
                model, train_data, loss_func, optimizer, args.device
            )
            logging.info(
                f'  - (Training)   Loss: {train_loss:8.5f}, '
                f'Accuracy: {100 * train_accu:3.3f}%, '
                f'Elapsed: {(time.time() - start):.2f}s'
            )

            # Evaluate on the validation set after each epoch
            start = time.time()
            validation_scores = self.test_epoch(model, valid_data, device=args.device)
            logging.info('  - (Validation)')
            self.log_scores(validation_scores)
            logging.info(f'  Validation time: {(time.time() - start) / 60:.2f} min')

            # Check for improvement on the validation set
            current_validation_score = sum(validation_scores.values())
            if current_validation_score > validation_history:
                logging.info('  Best validation score improved. Saving model...')
                validation_history = current_validation_score
                best_scores = validation_scores
                torch.save(model.state_dict(), args.model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info(f'  No improvement for {epochs_without_improvement} epoch(s).')

            # Early stopping
            if epochs_without_improvement >= self.patience:
                logging.info(f'  Early stopping triggered after {epoch_i + 1} epochs.')
                break

        # Final evaluation on the test set
        logging.info('\n  - (Final Test)')
        test_scores = self.test_epoch(model, test_data, device=args.device)
        self.log_scores(test_scores)

        logging.info("\n- (Finished!) \nBest validation scores:")
        self.log_scores(best_scores)

    def train_epoch(self, model, training_data, loss_func, optimizer, device):
        """
        Performs one epoch of training using standard full-precision.
        """
        model.train()

        total_loss = 0.0
        n_total_users = 0.0
        n_total_correct = 0.0

        for batch in tqdm(training_data, desc="  Training", ncols=100, leave=False):
            history_seq, history_seq_timestamp, history_seq_idx = (item.to(device) for item in batch)
            gold = history_seq[:, 1:]

            n_users = gold.ne(Constants.PAD).sum().item()
            if n_users == 0:
                continue
            n_total_users += n_users

            optimizer.zero_grad()

            # The 'autocast' context manager has been removed for a standard forward pass.
            loss, n_correct = model.get_performance(
                history_seq, history_seq_timestamp, history_seq_idx, loss_func, gold
            )

            # Standard backward pass and optimizer step.
            # The GradScaler logic has been replaced with these two lines.
            loss.backward()
            optimizer.step()

            optimizer.update_learning_rate()

            n_total_correct += n_correct.item()
            total_loss += loss.item() * n_users

        return total_loss / n_total_users, n_total_correct / n_total_users

    def test_epoch(self, model, validation_data, k_list=[10, 20, 50, 100], device=None):
        """
        Performs one epoch of evaluation using GPU-vectorized metrics to avoid CPU stalls.
        """
        model.eval()

        scores = {f'hits@{k}': 0.0 for k in k_list}
        scores.update({f'map@{k}': 0.0 for k in k_list})

        n_total_words = 0

        with torch.no_grad():
            for batch in tqdm(validation_data, desc="  Evaluating", ncols=100, leave=False):
                history_seq, history_seq_timestamp, history_seq_idx = (item.to(device) for item in batch)
                gold_flat = history_seq[:, 1:].contiguous().view(-1)
                valid_mask = gold_flat.ne(Constants.PAD)

                # Forward on GPU
                pred = model(history_seq, history_seq_timestamp, history_seq_idx)  # [N, |U|]

                n_valid = int(valid_mask.sum().item())
                if n_valid == 0:
                    continue

                for k in k_list:
                    topk_idx = torch.topk(pred, k=k, dim=1, largest=True, sorted=True).indices  # [N, k]

                    # hits@k
                    hits_any = topk_idx.eq(gold_flat.unsqueeze(1))
                    hits_any = hits_any & valid_mask.unsqueeze(1)
                    hits_sum = hits_any.any(dim=1).float().sum().item()
                    scores[f'hits@{k}'] += hits_sum

                    # map@k (single-label AP@k simplifies to reciprocal rank if present)
                    found_any = hits_any.any(dim=1)
                    first_pos = hits_any.float().argmax(dim=1)
                    reciprocal = torch.where(
                        found_any,
                        1.0 / (first_pos.float() + 1.0),
                        torch.zeros_like(first_pos, dtype=torch.float)
                    )
                    reciprocal = reciprocal.masked_select(valid_mask)
                    scores[f'map@{k}'] += reciprocal.sum().item()

                n_total_words += n_valid

        # Normalize scores by the total number of evaluation instances
        for k in k_list:
            scores[f'hits@{k}'] /= n_total_words
            scores[f'map@{k}'] /= n_total_words

        return scores

    @staticmethod
    def log_scores(scores):
        """Logs evaluation scores in a formatted table."""
        if not scores:
            return
        logging.info(f"  {'Metric':<12} | {'Score':<15}")
        logging.info(f"  {'-' * 12} | {'-' * 15}")
        for metric, score in scores.items():
            logging.info(f"  {metric:<12} | {score:<15.4f}")
