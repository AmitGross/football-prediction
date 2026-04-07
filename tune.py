# tune.py — Optuna hyperparameter search (50 trials, live output)
# Tunes: RF n_estimators/max_depth, XGB learning_rate/max_depth/subsample,
#        ELO_K, KALMAN_PROCESS_NOISE, KALMAN_MEASURE_NOISE
# Uses TimeSeriesSplit CV on 930 qualifier matches.
# Writes best params to best_params.json.

import json
import numpy as np
import pandas as pd
import optuna
import features as feat_module
from features import build_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from train import AveragingEnsemble

import argparse

DATA_PATH   = 'data/matches.csv'
N_SPLITS    = 5

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=50)
args, _ = parser.parse_known_args()
N_TRIALS = args.trials


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def objective(trial):
    # ── Hyperparameters to tune ────────────────────────────────────────────────
    # RF params
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 50,  500)
    rf_max_depth    = trial.suggest_int('rf_max_depth',     3,   12)
    # XGBoost params
    xgb_n_estimators   = trial.suggest_int('xgb_n_estimators',  100, 500)
    xgb_max_depth      = trial.suggest_int('xgb_max_depth',       3,  8)
    xgb_learning_rate  = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
    xgb_subsample      = trial.suggest_float('xgb_subsample',     0.6,  1.0)
    xgb_colsample      = trial.suggest_float('xgb_colsample',     0.6,  1.0)
    # Feature rating params
    elo_k         = trial.suggest_float('elo_k',         10.0, 50.0)
    process_noise = trial.suggest_float('process_noise',  1.0, 20.0)
    measure_noise = trial.suggest_float('measure_noise',  1.0, 20.0)

    # Inject hyperparams into features module globals
    feat_module.ELO_K                = elo_k
    feat_module.KALMAN_PROCESS_NOISE = process_noise
    feat_module.KALMAN_MEASURE_NOISE = measure_noise

    df = load_data()

    # Build full feature matrix once (walk-forward, no leakage)
    X, y_A, y_B = build_features(df)
    Y = np.column_stack([y_A, y_B])

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_tr, Y_val = Y[train_idx],      Y[val_idx]

        rf = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators = rf_n_estimators,
                max_depth    = rf_max_depth,
                random_state = 42,
                n_jobs       = -1,
            )
        )
        xgb = MultiOutputRegressor(
            XGBRegressor(
                objective        = 'count:poisson',
                n_estimators     = xgb_n_estimators,
                max_depth        = xgb_max_depth,
                learning_rate    = xgb_learning_rate,
                subsample        = xgb_subsample,
                colsample_bytree = xgb_colsample,
                random_state     = 42,
                n_jobs           = -1,
                verbosity        = 0,
            )
        )

        ensemble = AveragingEnsemble([rf, xgb])
        rf.fit(X_tr, Y_tr)
        xgb.fit(X_tr, Y_tr)
        preds = ensemble.predict(X_val)
        mae   = mean_absolute_error(Y_val, preds)
        mae_scores.append(mae)

    mean_mae = float(np.mean(mae_scores))
    return mean_mae


def callback(study, trial):
    print(
        f"  Trial {trial.number:>3d} | "
        f"MAE={trial.value:.4f} | "
        f"best={study.best_value:.4f} | "
        f"rf_est={trial.params['rf_n_estimators']:>4d}  "
        f"rf_d={trial.params['rf_max_depth']:>2d}  "
        f"xgb_est={trial.params['xgb_n_estimators']:>4d}  "
        f"xgb_d={trial.params['xgb_max_depth']:>2d}  "
        f"lr={trial.params['xgb_learning_rate']:.3f}  "
        f"elo_k={trial.params['elo_k']:.1f}  "
        f"proc={trial.params['process_noise']:.1f}  "
        f"meas={trial.params['measure_noise']:.1f}",
        flush=True
    )


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Starting Optuna search: {N_TRIALS} trials, {N_SPLITS}-fold TimeSeriesSplit")
    print("-" * 85)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    print("-" * 85)
    print(f"\nBest MAE : {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open('best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved to best_params.json")


if __name__ == '__main__':
    main()
