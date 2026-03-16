# nanoGPT Ablation Studies

Systematic ablation experiments on the Shakespeare character-level model to understand
the contribution of each architectural and training decision.

## Baseline
All ablations start from the `config/train_shakespeare_char.py` baseline:
- 6 layers, 6 heads, 384 embedding dim
- dropout=0.2, batch_size=64, block_size=256
- lr=1e-3, cosine schedule, 5000 iters

## Experiments

| Ablation | Variable | Values Tested | Hypothesis |
|----------|----------|--------------|------------|
| **A1: Dropout** | `dropout` | 0.0, 0.1, 0.2, 0.3 | Higher dropout hurts on small data (already overfitting) |
| **A2: Depth vs Width** | `n_layer`, `n_embd` | Iso-param configs | Deeper+narrower vs shallower+wider at same param count |
| **A3: Learning Rate** | `learning_rate` | 3e-4, 1e-3, 3e-3 | Sweet spot for baby GPT |
| **A4: Context Length** | `block_size` | 64, 128, 256, 512 | Longer context helps character-level modeling |
| **A5: Bias** | `bias` | True, False | GPTConfig notes "False is a bit better and faster" |
| **A6: Weight Decay** | `weight_decay` | 0.0, 0.01, 0.1, 1.0 | Regularization vs underfitting tradeoff |

## How to Run

```bash
# Run all ablations
python ablations/run_ablations.py

# Run a single ablation
python ablations/run_ablations.py --study dropout

# Run baseline only
python ablations/run_ablations.py --study baseline
```

## Analysis

Results are saved to `ablations/results/` as JSON files with:
- Final train/val loss for each config
- Training curves (loss at each eval interval)
- Wall-clock time per experiment
- Parameter counts

## Origin

These ablation studies were proposed by the Domain 5 multi-agent research system,
which analyzed the nanoGPT codebase and identified key architectural decisions
worth testing empirically.
