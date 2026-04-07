# evaluate.py
# Walk-forward evaluation on WC 2022 or WC 2026 matches.
#
# WC 2022 (actuals known — full retrospective):
#   python evaluate.py                          — frozen, all matches
#   python evaluate.py --limit 5               — frozen, first 5 matches
#   python evaluate.py --retrain               — retrain after each match (walk-forward)
#   python evaluate.py --retrain --limit 5
#
# WC 2026 (live — evaluates only rows where goals_A/goals_B are filled in):
#   python evaluate.py --year 2026             — frozen, all played matches so far
#   python evaluate.py --year 2026 --retrain   — walk-forward, all played matches
#
# Modes:
#   FROZEN  (default): model trained once on pre-tournament data, never retrained.
#   RETRAIN (--retrain): model is retrained after each result is known (live/walk-forward).

import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

import features
from predict import predict_match
from train import train, save_model, load_data

TRAIN_PATH = 'data/matches.csv'
_TEST_PATHS = {
    2022: 'data/wc2022.csv',
    2026: 'data/wc2026.csv',
}
TEMP_PATH  = 'data/matches_eval_tmp.csv'


def rps_single(p_win, p_draw, p_loss, actual_outcome):
    """
    Ranked Probability Score for one match (3 ordered outcomes: W, D, L).
    Lower is better. Perfect = 0.
    """
    o_win  = 1.0 if actual_outcome == 'win'  else 0.0
    o_draw = 1.0 if actual_outcome == 'draw' else 0.0
    cum_p1 = p_win
    cum_p2 = p_win + p_draw
    cum_o1 = o_win
    cum_o2 = o_win + o_draw
    return 0.5 * ((cum_p1 - cum_o1) ** 2 + (cum_p2 - cum_o2) ** 2)


def actual_outcome(goals_A, goals_B):
    if goals_A > goals_B:
        return 'win'
    elif goals_A == goals_B:
        return 'draw'
    else:
        return 'loss'


def evaluate(retrain=False, limit=None, year=2022):
    """
    Walk-forward evaluation against a WC fixture list.

    Parameters
    ----------
    retrain : bool
        False → frozen model (trained once before tournament).
        True  → walk-forward: retrain after every real result.
    limit   : int or None
        Evaluate only the first N matches that have actual results.
    year    : int (2022 or 2026)
        Which tournament to evaluate.
        2022 → wc2022.csv, Nov-2022 FIFA rankings (full actuals available).
        2026 → wc2026.csv, Apr-2026 FIFA rankings (evaluates only rows where
               goals_A / goals_B are filled in; prints a "no results yet"
               message if the tournament has not started).
    """
    features.set_fifa_rankings_year(year)

    TEST_PATH = _TEST_PATHS[year]
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
    test_df  = pd.read_csv(TEST_PATH,  parse_dates=['date'])
    test_df  = test_df.sort_values('date').reset_index(drop=True)

    # For 2026: only keep rows where actual results exist
    if year == 2026:
        test_df = test_df.dropna(subset=['goals_A', 'goals_B'])
        test_df = test_df.reset_index(drop=True)
        if len(test_df) == 0:
            print(f"\n[evaluate{year}] No results available yet in {TEST_PATH}.")
            print("Fill in goals_A / goals_B as matches are played, then re-run.")
            return

    test_df['goals_A'] = test_df['goals_A'].astype(int)
    test_df['goals_B'] = test_df['goals_B'].astype(int)

    if limit:
        test_df = test_df.head(limit)

    # For retrain mode: work on a temp CSV so matches.csv stays clean
    if retrain:
        shutil.copy(TRAIN_PATH, TEMP_PATH)

    history = train_df.copy()

    preds_A, preds_B     = [], []
    actuals_A, actuals_B = [], []
    correct_outcomes     = 0
    rps_scores           = []

    mode_label = f"{'RETRAIN' if retrain else 'FROZEN'} model — {len(test_df)} matches"
    col_w = 35
    print(f"\n{mode_label}")
    print(f"\n{'Match':<{col_w}} {'Pred':>6} {'Actual':>8}  OK")
    print("-" * (col_w + 25))

    for _, match in test_df.iterrows():
        team_A   = match['team_A']
        team_B   = match['team_B']
        actual_A = int(match['goals_A'])
        actual_B = int(match['goals_B'])

        result     = predict_match(team_A, team_B, history)
        pred_A     = result['goals_A']
        pred_B     = result['goals_B']
        pred_out   = result['outcome']
        actual_out = actual_outcome(actual_A, actual_B)
        ok         = pred_out == actual_out

        if ok:
            correct_outcomes += 1

        # RPS — requires probabilities from Poisson layer
        p_win_A_f = result.get('p_win_A', 0.0) / 100.0
        p_draw_f  = result.get('p_draw',  0.0) / 100.0
        p_win_B_f = result.get('p_win_B', 0.0) / 100.0
        rps_scores.append(rps_single(p_win_A_f, p_draw_f, p_win_B_f, actual_out))

        preds_A.append(pred_A)
        preds_B.append(pred_B)
        actuals_A.append(actual_A)
        actuals_B.append(actual_B)

        label    = f"{team_A} vs {team_B}"
        tick     = 'v' if ok else 'x'
        p_win_A  = result.get('p_win_A', '-')
        p_draw   = result.get('p_draw', '-')
        p_win_B  = result.get('p_win_B', '-')
        print(f"{label:<{col_w}} {pred_A}-{pred_B}  {actual_A}-{actual_B}  {tick}  W:{p_win_A:>5}% D:{p_draw:>5}% L:{p_win_B:>5}%")

        # Append real result to history
        new_row = pd.DataFrame([{
            'date':    match['date'],
            'team_A':  team_A,
            'team_B':  team_B,
            'goals_A': actual_A,
            'goals_B': actual_B,
        }])
        history = pd.concat([history, new_row], ignore_index=True)

        # Retrain mode: save updated history and retrain model
        if retrain:
            history.to_csv(TEMP_PATH, index=False)
            print(f"  [retraining on {len(history)} matches...]")
            trained_model, trained_features = train(history)
            save_model(trained_model, trained_features)

    # Clean up temp file
    if retrain:
        import os
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)


    n      = len(test_df)
    mae_A  = mean_absolute_error(actuals_A, preds_A)
    mae_B  = mean_absolute_error(actuals_B, preds_B)
    rmse_A = float(np.sqrt(np.mean((np.array(actuals_A) - np.array(preds_A)) ** 2)))
    rmse_B = float(np.sqrt(np.mean((np.array(actuals_B) - np.array(preds_B)) ** 2)))
    rmse   = float(np.sqrt(np.mean((np.array(actuals_A + actuals_B) - np.array(preds_A + preds_B)) ** 2)))
    acc    = correct_outcomes / n * 100
    mean_rps = float(np.mean(rps_scores))

    print("\n" + "=" * (col_w + 25))
    print(f"Matches evaluated : {n}")
    print(f"MAE  goals_A      : {mae_A:.4f}")
    print(f"MAE  goals_B      : {mae_B:.4f}")
    print(f"RMSE goals_A      : {rmse_A:.4f}")
    print(f"RMSE goals_B      : {rmse_B:.4f}")
    print(f"RMSE combined     : {rmse:.4f}   (benchmark: <1.65 strong, <1.75 good)")
    print(f"Outcome accuracy  : {correct_outcomes}/{n}  ({acc:.1f}%)  (benchmark: >0.52 good)")
    print(f"Mean RPS          : {mean_rps:.4f}   (benchmark: <0.21 solid, <0.20 strong, <0.195 excellent)")

    # Export results to Excel
    results_df = pd.DataFrame({
        'team_A': test_df['team_A'],
        'team_B': test_df['team_B'],
        'pred_goals_A': preds_A,
        'pred_goals_B': preds_B,
        'actual_goals_A': actuals_A,
        'actual_goals_B': actuals_B,
        'MAE_A': np.abs(np.array(preds_A) - np.array(actuals_A)),
        'MAE_B': np.abs(np.array(preds_B) - np.array(actuals_B)),
        'RMSE_A': (np.array(preds_A) - np.array(actuals_A)) ** 2,
        'RMSE_B': (np.array(preds_B) - np.array(actuals_B)) ** 2,
        'RPS': rps_scores
    })
    results_df['RMSE_A'] = np.sqrt(results_df['RMSE_A'])
    results_df['RMSE_B'] = np.sqrt(results_df['RMSE_B'])
    results_df['Outcome_Correct'] = [int(p == a) for p, a in zip(
        [ 'win' if pa > pb else 'draw' if pa == pb else 'loss' for pa, pb in zip(preds_A, preds_B) ],
        [ 'win' if aa > ab else 'draw' if aa == ab else 'loss' for aa, ab in zip(actuals_A, actuals_B) ]
    )]
    results_df['Mode'] = 'RETRAIN' if retrain else 'FROZEN'
    mode_tag   = 'retrain' if retrain else 'frozen'
    excel_name = f"results_wc{year}_{mode_tag}.xlsx"
    results_df.to_excel(excel_name, index=False)
    print(f"\nResults exported to {excel_name}\n")


if __name__ == '__main__':
    retrain = '--retrain' in sys.argv
    limit   = None
    year    = 2022
    if '--year' in sys.argv:
        idx  = sys.argv.index('--year')
        year = int(sys.argv[idx + 1])
    if '--limit' in sys.argv:
        idx   = sys.argv.index('--limit')
        limit = int(sys.argv[idx + 1])
    evaluate(retrain=retrain, limit=limit, year=year)
