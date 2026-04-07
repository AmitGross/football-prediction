# run_pipeline.py — Full pipeline orchestrator
#
# Usage:
#   python run_pipeline.py                        — tune (50) → train → evaluate (frozen)
#   python run_pipeline.py --skip-tune            — train → evaluate (frozen)
#   python run_pipeline.py --skip-tune --retrain  — train → evaluate (retrain mode)
#   python run_pipeline.py --trials 15            — tune (15) → train → evaluate
#   python run_pipeline.py --skip-tune --skip-train — just evaluate with existing model.pkl

import argparse
import subprocess
import sys
import time

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable] + cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {label} completed in {elapsed:.0f}s")

def main():
    parser = argparse.ArgumentParser(description='Football prediction pipeline')
    parser.add_argument('--skip-tune',   action='store_true', help='Skip Optuna tuning')
    parser.add_argument('--skip-train',  action='store_true', help='Skip training (use existing model.pkl)')
    parser.add_argument('--retrain',     action='store_true', help='Use retrain mode in evaluate.py')
    parser.add_argument('--trials',      type=int, default=50, help='Number of Optuna trials (default: 50)')
    args = parser.parse_args()

    steps = []

    if not args.skip_tune:
        steps.append((['tune.py', '--trials', str(args.trials)], f'Hyperparameter tuning ({args.trials} trials)'))

    if not args.skip_train:
        steps.append((['train.py'], 'Training model'))

    eval_cmd = ['evaluate.py']
    if args.retrain:
        eval_cmd.append('--retrain')
    steps.append((eval_cmd, f"Evaluating ({'retrain' if args.retrain else 'frozen'} mode)"))

    print(f"\nPipeline steps: {len(steps)}")
    for i, (_, label) in enumerate(steps, 1):
        print(f"  {i}. {label}")

    for cmd, label in steps:
        run(cmd, label)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
