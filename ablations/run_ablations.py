"""
Ablation study runner for nanoGPT.

Runs systematic experiments varying one hyperparameter at a time on the
Shakespeare character-level model, saving results for comparison.

Usage:
    python ablations/run_ablations.py                    # run all studies
    python ablations/run_ablations.py --study dropout    # run one study
    python ablations/run_ablations.py --study baseline   # baseline only
    python ablations/run_ablations.py --dry-run           # show configs without running

Proposed by Domain 5 multi-agent research system.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablations.configs import ALL_STUDIES, BASELINE


def run_experiment(name, config, dry_run=False):
    """Run a single training experiment with the given config overrides."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Show key config values
    keys_to_show = ['n_layer', 'n_head', 'n_embd', 'dropout', 'learning_rate',
                    'block_size', 'bias', 'weight_decay', 'batch_size', 'max_iters']
    for k in keys_to_show:
        if k in config:
            baseline_val = BASELINE.get(k)
            marker = " <-- CHANGED" if config[k] != baseline_val else ""
            print(f"  {k}: {config[k]}{marker}")

    if dry_run:
        print("  [DRY RUN - skipping training]")
        return None

    # Create output directory
    out_dir = config.get('out_dir', f'ablations/results/{name}')
    os.makedirs(out_dir, exist_ok=True)

    # Build command: python train.py with --key=value overrides
    # nanoGPT's configurator.py requires the -- prefix (line 31: assert arg.startswith('--'))
    cmd = [sys.executable, str(PROJECT_ROOT / 'train.py')]
    for key, value in config.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}={'True' if value else 'False'}")
        else:
            cmd.append(f"--{key}={value}")

    print(f"\n  Command: {' '.join(cmd[:3])} ...")

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per experiment
        )
        elapsed = time.time() - start_time

        # Parse final losses from output
        train_loss = None
        val_loss = None
        for line in result.stdout.split('\n'):
            if 'train loss' in line and 'val loss' in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == 'loss' and i > 0:
                        try:
                            if parts[i-1] == 'train':
                                train_loss = float(parts[i+1].rstrip(','))
                            elif parts[i-1] == 'val':
                                val_loss = float(parts[i+1].rstrip(','))
                        except (ValueError, IndexError):
                            pass

        # Save results
        experiment_result = {
            'name': name,
            'config': {k: v for k, v in config.items() if k != 'out_dir'},
            'train_loss': train_loss,
            'val_loss': val_loss,
            'elapsed_seconds': round(elapsed, 1),
            'return_code': result.returncode,
            'success': result.returncode == 0,
        }

        results_file = os.path.join(out_dir, 'result.json')
        with open(results_file, 'w') as f:
            json.dump(experiment_result, f, indent=2)

        if result.returncode == 0:
            print(f"\n  RESULT: train_loss={train_loss}, val_loss={val_loss}")
            print(f"  Time: {elapsed:.1f}s")
        else:
            print(f"\n  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")

        return experiment_result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n  TIMEOUT after {elapsed:.1f}s")
        return {'name': name, 'success': False, 'error': 'timeout'}


def run_study(study_name, configs, dry_run=False):
    """Run all experiments in a study."""
    print(f"\n{'#'*60}")
    print(f"  ABLATION STUDY: {study_name.upper()}")
    print(f"  Experiments: {len(configs)}")
    print(f"{'#'*60}")

    results = {}
    for exp_name, config in configs.items():
        result = run_experiment(f"{study_name}/{exp_name}", config, dry_run=dry_run)
        results[exp_name] = result

    # Print comparison table
    if not dry_run:
        print(f"\n{'='*60}")
        print(f"  RESULTS: {study_name.upper()}")
        print(f"{'='*60}")
        print(f"  {'Experiment':<25} {'Train Loss':>12} {'Val Loss':>12} {'Time':>8}")
        print(f"  {'-'*57}")
        for exp_name, result in results.items():
            if result and result.get('success'):
                tl = f"{result['train_loss']:.4f}" if result.get('train_loss') else 'N/A'
                vl = f"{result['val_loss']:.4f}" if result.get('val_loss') else 'N/A'
                t = f"{result['elapsed_seconds']:.0f}s"
                print(f"  {exp_name:<25} {tl:>12} {vl:>12} {t:>8}")
            else:
                print(f"  {exp_name:<25} {'FAILED':>12}")

    return results


def main():
    parser = argparse.ArgumentParser(description='nanoGPT Ablation Studies')
    parser.add_argument('--study', type=str, default='all',
                       choices=['all', 'baseline'] + list(ALL_STUDIES.keys()),
                       help='Which ablation study to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configs without running experiments')
    args = parser.parse_args()

    print("="*60)
    print("  nanoGPT ABLATION STUDIES")
    print("  Based on Shakespeare character-level model")
    print("="*60)

    if args.study == 'baseline':
        run_experiment('baseline', BASELINE, dry_run=args.dry_run)
    elif args.study == 'all':
        # Run baseline first
        run_experiment('baseline', BASELINE, dry_run=args.dry_run)
        # Then all studies
        for study_name, configs in ALL_STUDIES.items():
            run_study(study_name, configs, dry_run=args.dry_run)
    else:
        configs = ALL_STUDIES[args.study]
        run_study(args.study, configs, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("  ABLATION STUDIES COMPLETE")
    print("  Results saved to ablations/results/")
    print("="*60)


if __name__ == '__main__':
    main()
