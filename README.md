# ALTER Code and Data Supplement

This package contains the implementation of ALTER and the processed MIND data used in the experiments.

## Included Data

The complete processed datasets used in the paper are too large for the supplementary package. We therefore include MIND, the smallest of the three evaluated datasets, to provide a complete and directly executable example of the data pipeline, model training, and evaluation protocol. The same data interface and experimental procedure are used for X-Election and Zhihu.

The MIND files are located in `data/mind/`:

| File | Description |
| --- | --- |
| `user.txt` | User identifiers and available profile fields |
| `content.txt` | Content identifiers and content information |
| `cascades.txt` | Time ordered participation events for each content item |
| `user_history.txt` | Historical user and content interactions |

All data remain subject to the terms and conditions of the original MIND dataset.

## Code Structure

| Path | Description |
| --- | --- |
| `main.py` | Training and evaluation entry point |
| `models/alter.py` | Implementation of ALTER |
| `models/modules/` | Graph and sequence encoder components |
| `dataset/dataloader.py` | Data loading, chronological splitting, and sample construction |
| `trainer.py` | Optimization, checkpoint selection, and ranking evaluation |
| `utils.py` | Random seed, logging, and configuration utilities |

## Environment

The reported experiments use the following software environment:

- Python 3.10.19
- NumPy 1.26.4
- PyTorch 2.5.1 with CUDA 12.1
- PyTorch Lightning 2.6.0

CUDA is optional for code inspection, but a CUDA compatible GPU is recommended for training.

## Training on MIND

Run the following command from this directory:

```bash
python main.py \
  --model alter \
  --data mind \
  --data_root ./data \
  --split_strategy leave_one_out \
  --d_model 64 \
  --num_layers 2 \
  --num_heads 8 \
  --dropout 0.1 \
  --max_seq_len 128 \
  --max_time_bucket 32 \
  --alter_layers 3 \
  --alter_routing_anchors 8 \
  --alter_context_states 16 \
  --alter_assignment_budget 4 \
  --alter_temperature 2.0 \
  --alter_latent_regularization_weight 1e-6 \
  --epochs 1000 \
  --batch_size 1024 \
  --eval_batch_size 1024 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --num_negatives 100 \
  --patience 5 \
  --grad_clip 1.0 \
  --seed 42
```

The reported results are averaged over seeds `42`, `43`, and `44`. Repeat the command with the corresponding value of `--seed` for the other runs. The checkpoint with the highest validation NDCG@10 is selected automatically, and full candidate test evaluation is performed after training.

Training outputs, configurations, checkpoints, and metric logs are written under `runs/`.

## Evaluation Protocol

The code uses chronological leave one out splits. The observed participation graph is constructed only from training interactions. During training, each target is paired with 100 sampled valid negatives. During validation and testing, the target is ranked against all valid users after participants already observed in the cascade prefix are masked.
