# ablation.py
# Leave-one-GROUP-out ablation study for WC 2022 frozen model.
#
# Zeroes each feature group in turn, re-runs the full 64-match eval,
# and reports Î”accuracy / Î”RPS / Î”RMSE vs the all-features baseline.
#
# SAFETY: loads model_wc2022_v1.4.pkl directly â€” never reads or writes
# model.pkl, matches.csv, or any other active pipeline file.
#
# Usage:
#   python ablation.py
#
# Prerequisites:
#   python main.py train --year 2022   (creates model_wc2022_v1.4.pkl)
#
# Output:
#   ablation_wc2022.csv  â€” ranked table saved to disk

import os
import pickle
import numpy as np
import pandas as pd
from poisson import predict_from_lambdas, score_grid, result_probabilities
from ensemble import AveragingEnsemble  # needed for pickle deserialization
import features as _feat
from features import (
    EloRating, KalmanRating, calculate_form_features,
    calculate_h2h, calculate_days_rest,
    compute_pagerank_features, get_pagerank,
    calculate_neighbourhood_features,
    FORM_N,
)

MODEL_PATH = 'model_wc2022_v1.4.pkl'
TRAIN_PATH = 'data/matches.csv'
TEST_PATH  = 'data/wc2022.csv'

# â”€â”€ Feature groups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FEATURE_GROUPS = {
    'Elo': [
        'elo_A', 'elo_B', 'elo_diff',
    ],
    'Kalman': [
        'ka_atk', 'ka_def', 'ka_unc_atk', 'ka_unc_def',
        'kb_atk', 'kb_def', 'kb_unc_atk', 'kb_unc_def',
        'kalman_atk_diff', 'kalman_def_diff',
    ],
    'Form-5': [
        'wins_A', 'draws_A', 'losses_A', 'scored_A', 'conc_A', 'wscored_A', 'wconc_A',
        'wins_B', 'draws_B', 'losses_B', 'scored_B', 'conc_B', 'wscored_B', 'wconc_B',
    ],
    'Form-2': [
        'wins_A2', 'draws_A2', 'losses_A2', 'scored_A2', 'conc_A2', 'wscored_A2', 'wconc_A2',
        'wins_B2', 'draws_B2', 'losses_B2', 'scored_B2', 'conc_B2', 'wscored_B2', 'wconc_B2',
    ],
    'H2H': [
        'h2h_wins', 'h2h_draws', 'h2h_losses',
    ],
    'FIFA Rankings': [
        'fifa_rank_A', 'fifa_rank_B', 'fifa_rank_diff',
    ],
    'Rest & Match Count': [
        'rest_A', 'rest_B', 'match_count_A', 'match_count_B',
    ],
    'PageRank / HITS': [
        'win_pr_A',  'win_pr_B',  'win_pr_diff',
        'goal_pr_A', 'goal_pr_B', 'goal_pr_diff',
        'hub_A',     'hub_B',     'hub_diff',
        'auth_A',    'auth_B',    'auth_diff',
    ],
    'Neighbourhood Basic': [
        'opp_elo_A',      'opp_elo_B',      'opp_elo_diff',
        'opp_scored_A',   'opp_scored_B',
        'opp_conceded_A', 'opp_conceded_B',
        'n_opps_A',       'n_opps_B',
    ],
    'Neighbourhood Perf': [
        'weighted_opp_elo_A',        'weighted_opp_elo_B',        'weighted_opp_elo_diff',
        'win_rate_vs_top_A',         'win_rate_vs_top_B',         'win_rate_vs_top_diff',
        'avg_goal_diff_vs_opp_A',    'avg_goal_diff_vs_opp_B',    'avg_goal_diff_vs_opp_diff',
        'wtd_goal_diff_opp_A',       'wtd_goal_diff_opp_B',       'wtd_goal_diff_opp_diff',
    ],
}


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_model(path):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['features']


def _actual_outcome(goals_A, goals_B):
    if goals_A > goals_B:   return 'win'
    if goals_A == goals_B:  return 'draw'
    return 'loss'


def _rps(p_win, p_draw, p_loss, outcome):
    o_win = 1.0 if outcome == 'win'  else 0.0
    o_drw = 1.0 if outcome == 'draw' else 0.0
    return 0.5 * ((p_win - o_win)**2 + (p_win + p_draw - o_win - o_drw)**2)


def _build_row(team_A, team_B, history):
    """Build the feature dict for one match â€” mirrors predict_match() exactly."""
    history = history.sort_values('date').reset_index(drop=True)

    elo    = EloRating()
    kalman = KalmanRating()
    for _, r in history.iterrows():
        elo.update(r['team_A'], r['team_B'], r['goals_A'], r['goals_B'])
        kalman.update(r['team_A'], r['goals_A'], r['goals_B'])
        kalman.update(r['team_B'], r['goals_B'], r['goals_A'])

    elo_A, elo_B, elo_diff = elo.get_ratings(team_A, team_B)
    ka_atk, ka_def, ka_unc_atk, ka_unc_def, \
    kb_atk, kb_def, kb_unc_atk, kb_unc_def = kalman.get_ratings(team_A, team_B)

    wA  = calculate_form_features(history, team_A, FORM_N, elo.ratings)
    wB  = calculate_form_features(history, team_B, FORM_N, elo.ratings)
    w2A = calculate_form_features(history, team_A, 2,      elo.ratings)
    w2B = calculate_form_features(history, team_B, 2,      elo.ratings)

    h2h_wins, h2h_draws, h2h_losses = calculate_h2h(history, team_A, team_B)

    cur_date = history['date'].max() if len(history) > 0 else pd.Timestamp.now()
    rest_A = calculate_days_rest(history, team_A, cur_date)
    rest_B = calculate_days_rest(history, team_B, cur_date)

    pr   = compute_pagerank_features(history)
    pr_A = get_pagerank(pr, team_A)
    pr_B = get_pagerank(pr, team_B)

    nbr_A = calculate_neighbourhood_features(history, team_A, elo.ratings)
    nbr_B = calculate_neighbourhood_features(history, team_B, elo.ratings)

    def fifa(t): return _feat._FIFA_RANKINGS.get(t, _feat._FIFA_DEFAULT)

    return {
        'elo_A': elo_A, 'elo_B': elo_B, 'elo_diff': elo_diff,
        'ka_atk': ka_atk, 'ka_def': ka_def, 'ka_unc_atk': ka_unc_atk, 'ka_unc_def': ka_unc_def,
        'kb_atk': kb_atk, 'kb_def': kb_def, 'kb_unc_atk': kb_unc_atk, 'kb_unc_def': kb_unc_def,
        'kalman_atk_diff': ka_atk - kb_atk, 'kalman_def_diff': ka_def - kb_def,
        'wins_A': wA[0], 'draws_A': wA[1], 'losses_A': wA[2],
        'scored_A': wA[3], 'conc_A': wA[4], 'wscored_A': wA[5], 'wconc_A': wA[6],
        'wins_B': wB[0], 'draws_B': wB[1], 'losses_B': wB[2],
        'scored_B': wB[3], 'conc_B': wB[4], 'wscored_B': wB[5], 'wconc_B': wB[6],
        'wins_A2': w2A[0], 'draws_A2': w2A[1], 'losses_A2': w2A[2],
        'scored_A2': w2A[3], 'conc_A2': w2A[4], 'wscored_A2': w2A[5], 'wconc_A2': w2A[6],
        'wins_B2': w2B[0], 'draws_B2': w2B[1], 'losses_B2': w2B[2],
        'scored_B2': w2B[3], 'conc_B2': w2B[4], 'wscored_B2': w2B[5], 'wconc_B2': w2B[6],
        'h2h_wins': h2h_wins, 'h2h_draws': h2h_draws, 'h2h_losses': h2h_losses,
        'rest_A': rest_A, 'rest_B': rest_B,
        'match_count_A': int(((history['team_A'] == team_A) | (history['team_B'] == team_A)).sum()),
        'match_count_B': int(((history['team_A'] == team_B) | (history['team_B'] == team_B)).sum()),
        'fifa_rank_A': fifa(team_A), 'fifa_rank_B': fifa(team_B),
        'fifa_rank_diff': fifa(team_A) - fifa(team_B),
        'win_pr_A':  pr_A['win_pr'],  'win_pr_B':  pr_B['win_pr'],  'win_pr_diff':  pr_A['win_pr']  - pr_B['win_pr'],
        'goal_pr_A': pr_A['goal_pr'], 'goal_pr_B': pr_B['goal_pr'], 'goal_pr_diff': pr_A['goal_pr'] - pr_B['goal_pr'],
        'hub_A':     pr_A['hub'],     'hub_B':     pr_B['hub'],     'hub_diff':     pr_A['hub']     - pr_B['hub'],
        'auth_A':    pr_A['auth'],    'auth_B':    pr_B['auth'],    'auth_diff':    pr_A['auth']    - pr_B['auth'],
        'opp_elo_A':      nbr_A['avg_opp_elo'],       'opp_elo_B':      nbr_B['avg_opp_elo'],
        'opp_elo_diff':   nbr_A['avg_opp_elo']       - nbr_B['avg_opp_elo'],
        'opp_scored_A':   nbr_A['avg_opp_scored'],   'opp_scored_B':   nbr_B['avg_opp_scored'],
        'opp_conceded_A': nbr_A['avg_opp_conceded'], 'opp_conceded_B': nbr_B['avg_opp_conceded'],
        'n_opps_A':       nbr_A['n_opponents'],        'n_opps_B':       nbr_B['n_opponents'],
        'weighted_opp_elo_A':        nbr_A['weighted_opp_elo'],
        'weighted_opp_elo_B':        nbr_B['weighted_opp_elo'],
        'weighted_opp_elo_diff':     nbr_A['weighted_opp_elo']          - nbr_B['weighted_opp_elo'],
        'win_rate_vs_top_A':         nbr_A['win_rate_vs_top_teams'],
        'win_rate_vs_top_B':         nbr_B['win_rate_vs_top_teams'],
        'win_rate_vs_top_diff':      nbr_A['win_rate_vs_top_teams']     - nbr_B['win_rate_vs_top_teams'],
        'avg_goal_diff_vs_opp_A':    nbr_A['avg_goal_diff_vs_opp'],
        'avg_goal_diff_vs_opp_B':    nbr_B['avg_goal_diff_vs_opp'],
        'avg_goal_diff_vs_opp_diff': nbr_A['avg_goal_diff_vs_opp']     - nbr_B['avg_goal_diff_vs_opp'],
        'wtd_goal_diff_opp_A':       nbr_A['weighted_goal_diff_by_opp'],
        'wtd_goal_diff_opp_B':       nbr_B['weighted_goal_diff_by_opp'],
        'wtd_goal_diff_opp_diff':    nbr_A['weighted_goal_diff_by_opp'] - nbr_B['weighted_goal_diff_by_opp'],
    }


# â”€â”€ Core eval loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_eval(model, feature_cols, history_df, test_df, ablated_cols=None):
    """
    Run frozen eval over test_df.
    ablated_cols: iterable of column names to zero before model.predict().
    history_df is never mutated.
    """
    ablated = set(ablated_cols or [])
    history = history_df.copy()

    correct = 0
    rps_list, pred_As, pred_Bs, act_As, act_Bs = [], [], [], [], []

    for _, match in test_df.iterrows():
        team_A   = match['team_A']
        team_B   = match['team_B']
        actual_A = int(match['goals_A'])
        actual_B = int(match['goals_B'])

        row = _build_row(team_A, team_B, history)
        X   = pd.DataFrame([row])[feature_cols]

        for col in ablated:
            if col in X.columns:
                X[col] = 0.0

        raw   = model.predict(X)[0]
        lam_A = max(float(raw[0]), 0.0)
        lam_B = max(float(raw[1]), 0.0)

        display = predict_from_lambdas(lam_A, lam_B)
        pred_A  = display['goals_A']
        pred_B  = display['goals_B']

        grid              = score_grid(lam_A, lam_B)
        p_win, p_draw, p_loss = result_probabilities(grid)

        if p_win >= p_draw and p_win >= p_loss:    pred_out = 'win'
        elif p_draw >= p_win and p_draw >= p_loss: pred_out = 'draw'
        else:                                      pred_out = 'loss'

        actual_out = _actual_outcome(actual_A, actual_B)
        if pred_out == actual_out:
            correct += 1

        rps_list.append(_rps(p_win, p_draw, p_loss, actual_out))
        pred_As.append(pred_A); pred_Bs.append(pred_B)
        act_As.append(actual_A); act_Bs.append(actual_B)

    n  = len(test_df)
    pA = np.array(pred_As); pB = np.array(pred_Bs)
    aA = np.array(act_As);  aB = np.array(act_Bs)
    return {
        'accuracy': round(correct / n * 100, 1),
        'rps':      round(float(np.mean(rps_list)), 4),
        'rmse':     round(float(np.sqrt(np.mean(np.concatenate([(pA-aA)**2, (pB-aB)**2])))), 4),
    }


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def run_eval_retrain(initial_model, feature_cols, history_df, test_df, ablated_cols=None):
    """
    Walk-forward retrain eval: after each match result is revealed, append it
    to history and retrain the model before predicting the next match.
    Feature group is zeroed only at prediction time; retrain always uses all features.
    history_df is never mutated.
    """
    from train import train as _train_model

    ablated = set(ablated_cols or [])
    history = history_df.copy()
    model   = initial_model   # start from the pre-trained model

    correct = 0
    rps_list, pred_As, pred_Bs, act_As, act_Bs = [], [], [], [], []

    for _, match in test_df.iterrows():
        team_A   = match['team_A']
        team_B   = match['team_B']
        actual_A = int(match['goals_A'])
        actual_B = int(match['goals_B'])

        row = _build_row(team_A, team_B, history)
        X   = pd.DataFrame([row])[feature_cols]

        for col in ablated:
            if col in X.columns:
                X[col] = 0.0

        raw   = model.predict(X)[0]
        lam_A = max(float(raw[0]), 0.0)
        lam_B = max(float(raw[1]), 0.0)

        display = predict_from_lambdas(lam_A, lam_B)
        pred_A  = display['goals_A']
        pred_B  = display['goals_B']

        grid              = score_grid(lam_A, lam_B)
        p_win, p_draw, p_loss = result_probabilities(grid)

        if p_win >= p_draw and p_win >= p_loss:    pred_out = 'win'
        elif p_draw >= p_win and p_draw >= p_loss: pred_out = 'draw'
        else:                                      pred_out = 'loss'

        actual_out = _actual_outcome(actual_A, actual_B)
        if pred_out == actual_out:
            correct += 1

        rps_list.append(_rps(p_win, p_draw, p_loss, actual_out))
        pred_As.append(pred_A); pred_Bs.append(pred_B)
        act_As.append(actual_A); act_Bs.append(actual_B)

        # Append result and retrain for next prediction
        new_row = pd.DataFrame([{
            'date': match['date'], 'team_A': team_A, 'team_B': team_B,
            'goals_A': actual_A, 'goals_B': actual_B,
        }])
        history = pd.concat([history, new_row], ignore_index=True).sort_values('date').reset_index(drop=True)
        model, _ = _train_model(history)

    n  = len(test_df)
    pA = np.array(pred_As); pB = np.array(pred_Bs)
    aA = np.array(act_As);  aB = np.array(act_Bs)
    return {
        'accuracy': round(correct / n * 100, 1),
        'rps':      round(float(np.mean(rps_list)), 4),
        'rmse':     round(float(np.sqrt(np.mean(np.concatenate([(pA-aA)**2, (pB-aB)**2])))), 4),
    }


# WC 2022 group-stage cutoff (knockouts start Dec 3)
_GROUP_STAGE_END = '2022-12-02'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--knockout', action='store_true',
                        help='Ablate on knockout matches only (group stage pre-fed into history)')
    parser.add_argument('--retrain', action='store_true',
                        help='Walk-forward retrain after each revealed result (knockout mode only)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f'\nERROR: {MODEL_PATH} not found.')
        print('Run first:  python main.py train --year 2022')
        return

    print(f'Loading {MODEL_PATH} ...')
    model, feature_cols = _load_model(MODEL_PATH)

    _feat.set_fifa_rankings_year(2022)

    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
    all_wc   = pd.read_csv(TEST_PATH,  parse_dates=['date'])
    all_wc   = all_wc.sort_values('date').reset_index(drop=True)
    all_wc['goals_A'] = all_wc['goals_A'].astype(int)
    all_wc['goals_B'] = all_wc['goals_B'].astype(int)

    if args.knockout:
        # Group stage results pre-fed; only evaluate on knockouts
        group_df   = all_wc[all_wc['date'].dt.strftime('%Y-%m-%d') <= _GROUP_STAGE_END].copy()
        test_df    = all_wc[all_wc['date'].dt.strftime('%Y-%m-%d') >  _GROUP_STAGE_END].copy().reset_index(drop=True)
        history_df = pd.concat([train_df, group_df], ignore_index=True).sort_values('date').reset_index(drop=True)
        retrain_suffix = '-retrain' if args.retrain else ''
        mode_label = f'KNOCKOUT-ONLY{retrain_suffix.upper()} ({len(test_df)} matches, group stage pre-fed)'
        out_path   = f'ablation_wc2022_knockout{retrain_suffix}.csv'
    else:
        test_df    = all_wc
        history_df = train_df
        mode_label = f'FULL TOURNAMENT ({len(test_df)} matches)'
        out_path   = 'ablation_wc2022.csv'

    print(f'\nMode: {mode_label}')
    eval_fn = (run_eval_retrain if (args.knockout and args.retrain) else run_eval)
    print('Baseline (all features) ...')
    baseline = eval_fn(model, feature_cols, history_df, test_df)
    print(f"  Acc: {baseline['accuracy']}%  RPS: {baseline['rps']}  RMSE: {baseline['rmse']}\n")

    rows = []
    for group, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in feature_cols]
        print(f'  Ablating [{group}] ({len(present)} features) ...', end='', flush=True)
        m = eval_fn(model, feature_cols, history_df, test_df, ablated_cols=present)
        d_acc  = round(m['accuracy'] - baseline['accuracy'], 1)
        d_rps  = round(m['rps']      - baseline['rps'],      4)
        d_rmse = round(m['rmse']     - baseline['rmse'],     4)
        print(f"  Acc: {m['accuracy']}%  RPS: {m['rps']}  RMSE: {m['rmse']}  "
              f"dAcc: {d_acc:+.1f}%  dRPS: {d_rps:+.4f}")
        rows.append({'Group': group, 'N_features': len(present),
                     'Accuracy_%': m['accuracy'], 'RPS': m['rps'], 'RMSE': m['rmse'],
                     'dAcc_%': d_acc, 'dRPS': d_rps, 'dRMSE': d_rmse})

    rows.sort(key=lambda x: -x['dRPS'])

    W = 80
    print('\n' + '=' * W)
    print(f'ABLATION -- WC 2022 {mode_label}  (sorted by dRPS: most valuable first)')
    print(f"Baseline: Acc={baseline['accuracy']}%  RPS={baseline['rps']}  RMSE={baseline['rmse']}")
    print('=' * W)
    print(f"{'Group':<25} {'N':>3}  {'Acc%':>6}  {'RPS':>7}  {'RMSE':>7}  {'dAcc%':>7}  {'dRPS':>8}  {'dRMSE':>8}")
    print('-' * W)
    for r in rows:
        print(f"{r['Group']:<25} {r['N_features']:>3}  {r['Accuracy_%']:>6}  {r['RPS']:>7.4f}  "
              f"{r['RMSE']:>7.4f}  {r['dAcc_%']:>+7.1f}  {r['dRPS']:>+8.4f}  {r['dRMSE']:>+8.4f}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
