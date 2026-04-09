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

# ── Stage date boundaries ─────────────────────────────────────────────────────
# Maps each WC year to (stage_name, date_from, date_to) tuples in order.
# Matches are assigned to the first stage whose window contains their date.
_STAGE_WINDOWS = {
    2018: [
        ('Group Stage',  '2018-06-14', '2018-06-28'),
        ('Round of 16',  '2018-06-30', '2018-07-03'),
        ('Quarter-Final','2018-07-06', '2018-07-07'),
        ('Semi-Final',   '2018-07-10', '2018-07-11'),
        ('3rd Place',    '2018-07-14', '2018-07-14'),
        ('Final',        '2018-07-15', '2018-07-15'),
    ],
    2022: [
        ('Group Stage',  '2022-11-20', '2022-12-02'),
        ('Round of 16',  '2022-12-03', '2022-12-06'),
        ('Quarter-Final','2022-12-09', '2022-12-10'),
        ('Semi-Final',   '2022-12-13', '2022-12-14'),
        ('3rd Place',    '2022-12-17', '2022-12-17'),
        ('Final',        '2022-12-18', '2022-12-18'),
    ],
    2026: [
        ('Group Stage',  '2026-06-11', '2026-07-02'),
        ('Round of 32',  '2026-07-04', '2026-07-07'),
        ('Round of 16',  '2026-07-10', '2026-07-13'),
        ('Quarter-Final','2026-07-16', '2026-07-17'),
        ('Semi-Final',   '2026-07-20', '2026-07-21'),
        ('3rd Place',    '2026-07-24', '2026-07-24'),
        ('Final',        '2026-07-26', '2026-07-26'),
    ],
}

def _get_stage(date, year: int) -> str:
    """Return the tournament stage name for a given match date and WC year."""
    date_str = str(date)[:10]
    for stage, d_from, d_to in _STAGE_WINDOWS.get(year, []):
        if d_from <= date_str <= d_to:
            return stage
    return 'Unknown'


def _metrics_for_subset(pred_A, pred_B, act_A, act_B, rps):
    """Compute all loss metrics for a subset of matches. Returns a dict."""
    n = len(pred_A)
    if n == 0:
        return {'N': 0, 'MAE_A': None, 'MAE_B': None,
                'RMSE_A': None, 'RMSE_B': None, 'RMSE_combined': None,
                'Accuracy_%': None, 'Mean_RPS': None}
    pA, pB = np.array(pred_A), np.array(pred_B)
    aA, aB = np.array(act_A),  np.array(act_B)
    correct = sum(
        1 for pa, pb, aa, ab in zip(pA, pB, aA, aB)
        if (pa > pb) == (aa > ab) and (pa == pb) == (aa == ab)
    )
    return {
        'N':               n,
        'MAE_A':           float(mean_absolute_error(aA, pA)),
        'MAE_B':           float(mean_absolute_error(aB, pB)),
        'RMSE_A':          float(np.sqrt(np.mean((pA - aA) ** 2))),
        'RMSE_B':          float(np.sqrt(np.mean((pB - aB) ** 2))),
        'RMSE_combined':   float(np.sqrt(np.mean(np.concatenate([(pA-aA)**2, (pB-aB)**2])))),
        'Accuracy_%':      round(correct / n * 100, 1),
        'Mean_RPS':        round(float(np.mean(rps)), 4),
    }

MODEL_VERSION = 'v1.4'   # bump when features or model architecture changes

TRAIN_PATH = 'data/matches.csv'
_TEST_PATHS = {
    2018: 'data/wc2018.csv',
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

    # ── Build per-match results DataFrame ─────────────────────────────────────
    stages = [_get_stage(d, year) for d in test_df['date']]
    pred_outcomes   = ['win' if pa > pb else 'draw' if pa == pb else 'loss'
                       for pa, pb in zip(preds_A, preds_B)]
    actual_outcomes = ['win' if aa > ab else 'draw' if aa == ab else 'loss'
                       for aa, ab in zip(actuals_A, actuals_B)]

    results_df = pd.DataFrame({
        'Stage':          stages,
        'team_A':         test_df['team_A'].values,
        'team_B':         test_df['team_B'].values,
        'pred_goals_A':   preds_A,
        'pred_goals_B':   preds_B,
        'actual_goals_A': actuals_A,
        'actual_goals_B': actuals_B,
        'pred_outcome':   pred_outcomes,
        'actual_outcome': actual_outcomes,
        'Outcome_Correct':[int(p == a) for p, a in zip(pred_outcomes, actual_outcomes)],
        'MAE_A':          np.abs(np.array(preds_A) - np.array(actuals_A)),
        'MAE_B':          np.abs(np.array(preds_B) - np.array(actuals_B)),
        'RMSE_A':         np.sqrt((np.array(preds_A) - np.array(actuals_A)) ** 2),
        'RMSE_B':         np.sqrt((np.array(preds_B) - np.array(actuals_B)) ** 2),
        'RPS':            rps_scores,
        'Mode':           'RETRAIN' if retrain else 'FROZEN',
    })

    # ── Build Summary sheet ────────────────────────────────────────────────────
    # Define the breakdown groups we want rows for
    stage_order = ['Full Tournament', 'Group Stage', 'Knockout Rounds',
                   'Round of 32', 'Round of 16', 'Quarter-Final',
                   'Semi-Final', '3rd Place', 'Final']

    def _subset(label):
        if label == 'Full Tournament':
            return results_df
        elif label == 'Knockout Rounds':
            return results_df[results_df['Stage'] != 'Group Stage']
        else:
            return results_df[results_df['Stage'] == label]

    summary_rows = []
    for label in stage_order:
        sub = _subset(label)
        if len(sub) == 0:
            continue
        m = _metrics_for_subset(
            sub['pred_goals_A'].tolist(), sub['pred_goals_B'].tolist(),
            sub['actual_goals_A'].tolist(), sub['actual_goals_B'].tolist(),
            sub['RPS'].tolist(),
        )
        summary_rows.append({'Stage': label, **m})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(0, 'Model', f"WC {year} — {'RETRAIN (walk-forward)' if retrain else 'FROZEN'}")
    summary_df.insert(1, 'Version', MODEL_VERSION)

    # ── Write multi-sheet Excel ────────────────────────────────────────────────
    mode_tag   = 'retrain' if retrain else 'frozen'
    excel_name = f"results_wc{year}_{mode_tag}_{MODEL_VERSION}.xlsx"

    with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Predictions', index=False)

    print(f"\nResults exported to {excel_name}")
    print(f"  Sheet 'Summary'     — metrics by stage ({len(summary_df)} rows)")
    print(f"  Sheet 'Predictions' — per-match detail ({len(results_df)} rows)\n")


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
