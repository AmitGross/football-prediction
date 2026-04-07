# calibrate.py — Isotonic probability calibration for W/D/L outcomes
#
# Strategy: time-split calibration
#   1. Build walk-forward features for the full training set (no leakage).
#   2. Use the LAST 25 % of rows as the calibration set (the model was trained
#      on 100 % so it has seen these matches, but isotonic calibration learns
#      the monotone mapping from model-predicted probs to empirical frequencies,
#      which is still corrective even when slightly in-sample).
#   3. Fit three independent IsotonicRegression models (one per outcome).
#   4. At inference: apply all three, then renormalise.

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from features import build_features
from poisson import score_grid, result_probabilities

CALIBRATOR_PATH = 'calibrator.pkl'
CALIB_SPLIT     = 0.75    # use last 25 % for calibration


# ── Inference helper (used by predict.py) ─────────────────────────────────────

def apply_calibration(calibrator, p_win, p_draw, p_loss):
    """
    Apply isotonic calibration to raw W/D/L probabilities, then renormalise.
    Returns (p_win, p_draw, p_loss) as floats in [0, 1].
    Safe to call with calibrator=None (returns inputs unchanged).
    """
    if calibrator is None:
        return p_win, p_draw, p_loss
    c_win  = float(calibrator['win'].predict( [p_win])[0])
    c_draw = float(calibrator['draw'].predict([p_draw])[0])
    c_loss = float(calibrator['loss'].predict([p_loss])[0])
    total  = c_win + c_draw + c_loss
    if total < 1e-6:
        return 1 / 3, 1 / 3, 1 / 3
    return c_win / total, c_draw / total, c_loss / total


def load_calibrator(path=CALIBRATOR_PATH):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Fitting ───────────────────────────────────────────────────────────────────

def _collect_calibration_probs(df, model, feature_cols):
    """
    Build walk-forward features for all rows, then predict λ with the given model
    for the calibration slice (last 25 %). Returns arrays of Poisson W/D/L probs
    and actual outcomes (0=W, 1=D, 2=L).
    """
    X, y_A, y_B = build_features(df)
    X = X[feature_cols]

    split  = int(len(X) * CALIB_SPLIT)
    X_c    = X.iloc[split:].values
    yA_c   = y_A.iloc[split:].values
    yB_c   = y_B.iloc[split:].values

    preds = model.predict(X_c)

    p_wins, p_draws, p_losses, outcomes = [], [], [], []
    for i in range(len(preds)):
        lam_A = max(float(preds[i, 0]), 1e-6)
        lam_B = max(float(preds[i, 1]), 1e-6)
        grid  = score_grid(lam_A, lam_B)
        pw, pd_, pl = result_probabilities(grid)
        p_wins.append(pw)
        p_draws.append(pd_)
        p_losses.append(pl)
        gA, gB = int(yA_c[i]), int(yB_c[i])
        outcomes.append(0 if gA > gB else (1 if gA == gB else 2))

    return (
        np.array(p_wins),
        np.array(p_draws),
        np.array(p_losses),
        np.array(outcomes, dtype=int),
    )


def run_calibration(train_path='data/matches.csv', calibrator_path=CALIBRATOR_PATH):
    """
    Fit and save the probability calibrator.
    Requires model.pkl to already exist (run python main.py train first).
    """
    if not os.path.exists('model.pkl'):
        raise FileNotFoundError("model.pkl not found — run 'python main.py train' first")

    with open('model.pkl', 'rb') as f:
        payload = pickle.load(f)
    model, feature_cols = payload['model'], payload['features']

    df = pd.read_csv(train_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    print(f"Building calibration features on {len(df)} training matches ...")

    p_wins, p_draws, p_losses, outcomes = _collect_calibration_probs(df, model, feature_cols)
    n = len(outcomes)
    print(f"Calibration slice  : {n} matches (last {100 * (1 - CALIB_SPLIT):.0f}% of training data)")

    y_win  = (outcomes == 0).astype(float)
    y_draw = (outcomes == 1).astype(float)
    y_loss = (outcomes == 2).astype(float)

    cal_win  = IsotonicRegression(out_of_bounds='clip').fit(p_wins,   y_win)
    cal_draw = IsotonicRegression(out_of_bounds='clip').fit(p_draws,  y_draw)
    cal_loss = IsotonicRegression(out_of_bounds='clip').fit(p_losses, y_loss)

    calibrator = {'win': cal_win, 'draw': cal_draw, 'loss': cal_loss}
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {calibrator_path}")

    # Sanity print
    print(f"  Base rates (actual) — "
          f"Win:{y_win.mean():.3f}  Draw:{y_draw.mean():.3f}  Loss:{y_loss.mean():.3f}")
    print(f"  Mean probs (model)  — "
          f"Win:{p_wins.mean():.3f}  Draw:{p_draws.mean():.3f}  Loss:{p_losses.mean():.3f}")
    return calibrator


if __name__ == '__main__':
    run_calibration()
