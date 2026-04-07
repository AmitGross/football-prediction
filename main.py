# main.py — Unified entry point for the football prediction pipeline
#
# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
#
#   python main.py train                          Train all models
#   python main.py tune [--trials N]              Hyperparameter search
#   python main.py evaluate [--retrain] [--limit N]      WC 2022 evaluation
#   python main.py evaluate2026 [--retrain] [--limit N]  WC 2026 live eval
#   python main.py simulate2026                   Full 2026 tournament simulation
#   python main.py predict "France" "Brazil"      Predict a single match
#
# ──────────────────────────────────────────────────────────────────────────────
# evaluate  — RETROSPECTIVE EVALUATION (WC 2022, all actuals known)
# ──────────────────────────────────────────────────────────────────────────────
#   Compares predictions to actual scores. Reports RMSE, MAE, RPS, accuracy.
#   Exports to results_wc2022_frozen.xlsx or results_wc2022_retrain.xlsx.
#
#   Modes:
#     FROZEN  (default): model trained once on pre-tournament data, frozen.
#                        → python main.py evaluate
#     RETRAIN (--retrain): retrain after every real result (walk-forward).
#                        → python main.py evaluate --retrain
#
# ──────────────────────────────────────────────────────────────────────────────
# evaluate2026  — LIVE EVALUATION (WC 2026, as results come in)
# ──────────────────────────────────────────────────────────────────────────────
#   Same as evaluate but targets data/wc2026.csv with Apr-2026 FIFA rankings.
#   Automatically skips rows where goals_A / goals_B are not yet filled in.
#   Before the tournament starts, it will print "No results available yet".
#
#   Workflow once the tournament starts:
#     1. Fill in goals_A / goals_B in data/wc2026.csv after each match.
#     2. Run `python main.py evaluate2026 --retrain` to update the model and
#        get latest metrics.
#   Exports to results_wc2026_frozen.xlsx or results_wc2026_retrain.xlsx.
#
#     FROZEN  → python main.py evaluate2026
#     RETRAIN → python main.py evaluate2026 --retrain
#
# ──────────────────────────────────────────────────────────────────────────────
# simulate2026  — PRE-TOURNAMENT SIMULATION (WC 2026, no actual results yet)
# ──────────────────────────────────────────────────────────────────────────────
#   Use this BEFORE the tournament starts, when no actual results are available.
#
#   Predicts the full tournament from scratch using the frozen model:
#     1. Predicts all group stage matches
#     2. Computes standings from predicted scores to determine who qualifies
#     3. Simulates all knockout rounds (R32 → R16 → QF → SF → Final)
#
#   No loss metrics are computed (no actuals to compare against).
#   Exports to predictions_wc2026_full.xlsx with group standings + knockout bracket.
#
#   NOTE: This is a "what-if" simulation only. Once the tournament starts and
#         real results are available, switch to `evaluate --retrain` instead.
#         The knockout bracket in this simulation is derived entirely from the
#         model's own group stage predictions — it is self-consistent but
#         hypothetical.
#
#   Uses the FROZEN model (trained on all pre-2026 data). No retraining during
#   simulation, since there are no real results to retrain on.
#         → python main.py simulate2026
#
# ──────────────────────────────────────────────────────────────────────────────
# Training order (main.py train):
#   1. λ regressor   (RF + XGBoost)        → model.pkl
#   2. Outcome classifier (XGBClassifier)  → classifier.pkl
#   3. Dixon-Coles MLE ratings             → dc_ratings.pkl
#   4. Isotonic probability calibrator     → calibrator.pkl
#
# Graph features: networkx graphs where nodes are countries, edges are matches.
# See features.py for details.
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import pandas as pd

DATA_PATH = 'data/matches.csv'


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_train(args):
    import features
    features.set_fifa_rankings_year(2026)  # train with latest 2026 rankings
    from train import load_data, train, save_model, train_classifier
    from dc_ratings import fit_dc_ratings
    from calibrate import run_calibration

    df = load_data(DATA_PATH)

    print("\n=== [1/4] Training λ regressor (RF + XGBoost) ===")
    model, features = train(df)
    save_model(model, features)

    print("\n=== [2/4] Training outcome classifier (XGBoost W/D/L) ===")
    train_classifier(df)

    print("\n=== [3/4] Fitting Dixon-Coles MLE strength ratings ===")
    fit_dc_ratings(df)

    print("\n=== [4/4] Fitting isotonic probability calibrator ===")
    run_calibration()

    print("\nDone. All models saved.")
    print("  model.pkl          λ regressor")
    print("  classifier.pkl     W/D/L outcome classifier")
    print("  dc_ratings.pkl     Dixon-Coles team strength ratings")
    print("  calibrator.pkl     Isotonic probability calibrator")
    print("\nNext: python main.py evaluate")


def cmd_tune(args):
    import subprocess, sys
    cmd = [sys.executable, 'tune.py', '--trials', str(args.trials)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    print("\nTuning complete. Run 'python main.py train' to retrain with the new params.")


def cmd_evaluate(args):
    from evaluate import evaluate
    evaluate(retrain=args.retrain, limit=args.limit, year=2022)


def cmd_evaluate2026(args):
    from evaluate import evaluate
    evaluate(retrain=args.retrain, limit=args.limit, year=2026)


def cmd_simulate2026(args):
    import features
    features.set_fifa_rankings_year(2026)  # WC 2026 uses Apr 2026 rankings
    import subprocess, sys
    subprocess.run([sys.executable, 'simulate_wc2026.py'])


def cmd_predict(args):
    import features
    features.set_fifa_rankings_year(2026)  # single-match predict uses latest rankings
    df     = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df     = df.sort_values('date').reset_index(drop=True)

    from predict import predict_match
    result = predict_match(args.team_a, args.team_b, df)

    width = max(len(args.team_a), len(args.team_b)) + 2
    print(f"\n{'Match':<{width}}  {args.team_a} vs {args.team_b}")
    print(f"  Predicted score : {result['goals_A']}-{result['goals_B']}")
    print(f"  Win  {result['p_win_A']:>5.1f}%   "
          f"Draw  {result['p_draw']:>5.1f}%   "
          f"Loss  {result['p_win_B']:>5.1f}%")
    print(f"  Expected goals  : λ_A={result['lam_A']:.2f}  λ_B={result['lam_B']:.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Football prediction pipeline',
    )
    sub = parser.add_subparsers(dest='command')

    # train
    sub.add_parser('train', help='Train all models (regressor + classifier + DC ratings + calibrator)')

    # tune
    tune_p = sub.add_parser('tune', help='Hyperparameter search with Optuna')
    tune_p.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')

    # evaluate (WC 2022)
    eval_p = sub.add_parser(
        'evaluate',
        help='Retrospective evaluation on WC 2022 (all actuals known). FROZEN or --retrain.')
    eval_p.add_argument('--retrain', action='store_true',
                        help='Walk-forward: retrain after each result. Default: frozen.')
    eval_p.add_argument('--limit', type=int, default=None,
                        help='Only evaluate first N matches')

    # evaluate2026 (WC 2026 live)
    eval26_p = sub.add_parser(
        'evaluate2026',
        help='Live evaluation on WC 2026 as results come in. Fill goals in wc2026.csv first.')
    eval26_p.add_argument('--retrain', action='store_true',
                          help='Walk-forward: retrain after each result. Default: frozen.')
    eval26_p.add_argument('--limit', type=int, default=None,
                          help='Only evaluate first N played matches')

    # simulate2026
    sub.add_parser(
        'simulate2026',
        help=(
            'Pre-tournament simulation for WC 2026 (no actual results). '
            'Predicts full tournament: group stage → standings → knockouts → champion. '
            'Uses frozen model. Switch to `evaluate --retrain` once the tournament starts.'
        ))

    # predict
    pred_p = sub.add_parser('predict', help='Predict a single match')
    pred_p.add_argument('team_a', help='Home / first team name')
    pred_p.add_argument('team_b', help='Away / second team name')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'tune':
        cmd_tune(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'evaluate2026':
        cmd_evaluate2026(args)
    elif args.command == 'simulate2026':
        cmd_simulate2026(args)
    elif args.command == 'predict':
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
