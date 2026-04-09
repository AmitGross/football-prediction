# train.py

import json
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor, XGBClassifier
import features as feat_module
# FIFA rankings year is set by the caller (main.py train --year) before importing
from features import build_features
from ensemble import AveragingEnsemble

PARAMS_PATH     = 'best_params.json'
MODEL_VERSION   = 'v1.4'   # bump when features or model architecture changes


def _load_best_params():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return {}


def _apply_feature_params(p):
    if 'elo_k' in p:
        feat_module.ELO_K = p['elo_k']
    if 'process_noise' in p:
        feat_module.KALMAN_PROCESS_NOISE = p['process_noise']
    if 'measure_noise' in p:
        feat_module.KALMAN_MEASURE_NOISE = p['measure_noise']


DATA_PATH       = 'data/matches.csv'
MODEL_PATH      = 'model.pkl'
CLASSIFIER_PATH = 'classifier.pkl'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df)} matches.")
    return df


def train(df):
    p = _load_best_params()
    _apply_feature_params(p)

    print("Building features...")
    X, y_goals_A, y_goals_B = build_features(df)
    Y = np.column_stack([y_goals_A, y_goals_B])

    print(f"Feature matrix: {X.shape}")

    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators = p.get('rf_n_estimators', 200),
            max_depth    = p.get('rf_max_depth', 6),
            random_state = 42,
            n_jobs       = -1
        )
    )

    xgb = MultiOutputRegressor(
        XGBRegressor(
            objective        = 'count:poisson',
            n_estimators     = p.get('xgb_n_estimators', 300),
            max_depth        = p.get('xgb_max_depth', 5),
            learning_rate    = p.get('xgb_learning_rate', 0.05),
            subsample        = p.get('xgb_subsample', 0.8),
            colsample_bytree = p.get('xgb_colsample', 0.8),
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
    )

    print("Training Random Forest...")
    rf.fit(X, Y)
    print("Training XGBoost...")
    xgb.fit(X, Y)

    model = AveragingEnsemble([rf, xgb])
    print("Ensemble trained (RF + XGBoost).")
    return model, X.columns.tolist()


def save_model(model, feature_cols, path=MODEL_PATH):
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols, 'version': MODEL_VERSION}, f)
    with open('model_version.txt', 'w') as f:
        f.write(f"{MODEL_VERSION} | {len(feature_cols)} features\n")
    print(f"Model saved to {path} [{MODEL_VERSION}, {len(feature_cols)} features]")


def retrain_after_match(team_A, team_B, goals_A, goals_B, date,
                        data_path=DATA_PATH, model_path=MODEL_PATH):
    """
    Called after every real match result is known.
    1. Appends the new match to the CSV
    2. Retrains the model from scratch on all data
    3. Saves the new model.pkl
    """
    df = load_data(data_path)

    new_row = pd.DataFrame([{
        'date':    date,
        'team_A':  team_A,
        'team_B':  team_B,
        'goals_A': goals_A,
        'goals_B': goals_B
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(data_path, index=False)
    print(f"Match appended: {team_A} {goals_A}-{goals_B} {team_B}")

    model, features = train(df)
    save_model(model, features, model_path)


def train_classifier(df, classifier_path=CLASSIFIER_PATH):
    """
    Train an XGBClassifier (W/D/L) on the same features as the λ regressor.
    outcome labels: 0=win_A, 1=draw, 2=loss_A
    Saved to classifier.pkl as {'model': clf, 'features': feature_cols}.
    """
    p = _load_best_params()
    _apply_feature_params(p)

    print("Building features for classifier...")
    X, y_goals_A, y_goals_B = build_features(df)
    y_outcome = np.where(
        y_goals_A.values > y_goals_B.values, 0,
        np.where(y_goals_A.values == y_goals_B.values, 1, 2)
    )

    clf = XGBClassifier(
        objective        = 'multi:softprob',
        num_class        = 3,
        n_estimators     = p.get('xgb_n_estimators', 300),
        max_depth        = p.get('xgb_max_depth', 5),
        learning_rate    = p.get('xgb_learning_rate', 0.05),
        subsample        = p.get('xgb_subsample', 0.8),
        colsample_bytree = p.get('xgb_colsample', 0.8),
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
        eval_metric      = 'mlogloss',
    )
    print("Training outcome classifier (XGBoost W/D/L) ...")
    clf.fit(X.values, y_outcome)

    payload = {'model': clf, 'features': X.columns.tolist()}
    with open(classifier_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Classifier saved to {classifier_path}")
    return clf, X.columns.tolist()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026, choices=[2022, 2026],
                        help='WC year — controls FIFA rankings snapshot (default: 2026)')
    _args = parser.parse_args()
    feat_module.set_fifa_rankings_year(_args.year)
    print(f'Using FIFA {_args.year} rankings.')
    df            = load_data()
    model, features = train(df)
    save_model(model, features)