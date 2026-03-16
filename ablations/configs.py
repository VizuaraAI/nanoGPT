"""
Ablation study configurations for nanoGPT Shakespeare character-level model.

Each ablation varies ONE parameter from the baseline while keeping everything
else constant, following standard ML ablation methodology.

Proposed by Domain 5 multi-agent research system analysis of nanoGPT.
"""

# =============================================================================
# BASELINE CONFIG (matches config/train_shakespeare_char.py)
# =============================================================================
BASELINE = dict(
    dataset='shakespeare_char',
    out_dir='ablations/results/baseline',
    eval_interval=250,
    eval_iters=200,
    log_interval=10,
    always_save_checkpoint=False,
    wandb_log=False,
    gradient_accumulation_steps=1,
    batch_size=64,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    learning_rate=1e-3,
    max_iters=5000,
    lr_decay_iters=5000,
    min_lr=1e-4,
    beta2=0.99,
    warmup_iters=100,
    bias=True,
    weight_decay=0.1,
)

# =============================================================================
# A1: DROPOUT ABLATION
# Hypothesis: dropout=0.2 is already tuned for this small dataset.
# Lower dropout may underfit less, higher may regularize too aggressively.
# =============================================================================
A1_DROPOUT = {
    'dropout_0.0': dict(**BASELINE, dropout=0.0, out_dir='ablations/results/a1_dropout/drop_0.0'),
    'dropout_0.1': dict(**BASELINE, dropout=0.1, out_dir='ablations/results/a1_dropout/drop_0.1'),
    'dropout_0.2': dict(**BASELINE, dropout=0.2, out_dir='ablations/results/a1_dropout/drop_0.2'),
    'dropout_0.3': dict(**BASELINE, dropout=0.3, out_dir='ablations/results/a1_dropout/drop_0.3'),
}

# =============================================================================
# A2: DEPTH VS WIDTH (ISO-PARAMETER)
# All configs have roughly the same parameter count (~10.6M).
# Tests whether depth or width matters more for character-level LM.
#
# Parameter count formula (approximate):
#   params ≈ 12 * n_layer * n_embd^2 (dominated by attention + MLP weights)
#
# Configs:
#   Deep+Narrow:  12 layers, 4 heads, 288 embd → ~12*12*288^2 ≈ 11.9M
#   Baseline:      6 layers, 6 heads, 384 embd → ~12*6*384^2  ≈ 10.6M
#   Shallow+Wide:  3 layers, 8 heads, 512 embd → ~12*3*512^2  ≈  9.4M
# =============================================================================
A2_DEPTH_WIDTH = {
    'deep_narrow':   dict(**BASELINE, n_layer=12, n_head=4, n_embd=288,
                          out_dir='ablations/results/a2_depth_width/deep_narrow'),
    'baseline':      dict(**BASELINE,
                          out_dir='ablations/results/a2_depth_width/baseline'),
    'shallow_wide':  dict(**BASELINE, n_layer=3, n_head=8, n_embd=512,
                          out_dir='ablations/results/a2_depth_width/shallow_wide'),
}

# =============================================================================
# A3: LEARNING RATE
# The baseline uses 1e-3 which is noted as "with baby networks can afford
# to go a bit higher". Test the boundaries.
# =============================================================================
A3_LEARNING_RATE = {
    'lr_3e-4': dict(**BASELINE, learning_rate=3e-4, min_lr=3e-5,
                     out_dir='ablations/results/a3_lr/lr_3e-4'),
    'lr_1e-3': dict(**BASELINE, learning_rate=1e-3, min_lr=1e-4,
                     out_dir='ablations/results/a3_lr/lr_1e-3'),
    'lr_3e-3': dict(**BASELINE, learning_rate=3e-3, min_lr=3e-4,
                     out_dir='ablations/results/a3_lr/lr_3e-3'),
}

# =============================================================================
# A4: CONTEXT LENGTH (BLOCK SIZE)
# Character-level models may benefit from longer context since individual
# characters carry less information than tokens. But longer context is
# more expensive (quadratic attention) and may not help on small data.
# =============================================================================
A4_BLOCK_SIZE = {
    'ctx_64':  dict(**BASELINE, block_size=64,
                     out_dir='ablations/results/a4_block_size/ctx_64'),
    'ctx_128': dict(**BASELINE, block_size=128,
                     out_dir='ablations/results/a4_block_size/ctx_128'),
    'ctx_256': dict(**BASELINE, block_size=256,
                     out_dir='ablations/results/a4_block_size/ctx_256'),
    'ctx_512': dict(**BASELINE, block_size=512,
                     out_dir='ablations/results/a4_block_size/ctx_512'),
}

# =============================================================================
# A5: BIAS ABLATION
# GPTConfig notes: "True: bias in Linears and LayerNorms, like GPT-2.
# False: a bit better and faster". Let's test this claim.
# =============================================================================
A5_BIAS = {
    'bias_true':  dict(**BASELINE, bias=True,
                        out_dir='ablations/results/a5_bias/bias_true'),
    'bias_false': dict(**BASELINE, bias=False,
                        out_dir='ablations/results/a5_bias/bias_false'),
}

# =============================================================================
# A6: WEIGHT DECAY
# AdamW weight decay is a key regularization knob. The baseline uses 0.1
# (standard for transformers). Test extremes.
# =============================================================================
A6_WEIGHT_DECAY = {
    'wd_0.0':  dict(**BASELINE, weight_decay=0.0,
                     out_dir='ablations/results/a6_weight_decay/wd_0.0'),
    'wd_0.01': dict(**BASELINE, weight_decay=0.01,
                     out_dir='ablations/results/a6_weight_decay/wd_0.01'),
    'wd_0.1':  dict(**BASELINE, weight_decay=0.1,
                     out_dir='ablations/results/a6_weight_decay/wd_0.1'),
    'wd_1.0':  dict(**BASELINE, weight_decay=1.0,
                     out_dir='ablations/results/a6_weight_decay/wd_1.0'),
}

# =============================================================================
# ALL STUDIES
# =============================================================================
ALL_STUDIES = {
    'dropout': A1_DROPOUT,
    'depth_width': A2_DEPTH_WIDTH,
    'learning_rate': A3_LEARNING_RATE,
    'block_size': A4_BLOCK_SIZE,
    'bias': A5_BIAS,
    'weight_decay': A6_WEIGHT_DECAY,
}
