# predict.py

import pandas as pd
import pickle
import numpy as np
from poisson import predict_from_lambdas, score_grid, result_probabilities
from ensemble import AveragingEnsemble  # needed for pickle to deserialise model.pkl
from features import (
    EloRating, KalmanRating, calculate_form_features,
    calculate_h2h, calculate_days_rest,
    build_graph, compute_pagerank_features, get_pagerank,
    calculate_neighbourhood_features,
    FORM_N, _FIFA_RANKINGS, _FIFA_DEFAULT
)

MODEL_PATH = 'model.pkl'


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['features']


def predict_match(team_A, team_B, current_matches):
    """
    Predict goals for team_A and team_B.
    Derives outcome from predicted scores.
    Returns: { goals_A, goals_B, outcome }
    """
    model, feature_cols = load_model()

    current_matches = current_matches.sort_values('date').reset_index(drop=True)

    # Replay all past matches to build current Elo and Kalman state
    elo_system    = EloRating()
    kalman_system = KalmanRating()
    for _, row in current_matches.iterrows():
        elo_system.update(row['team_A'], row['team_B'], row['goals_A'], row['goals_B'])
        kalman_system.update(row['team_A'], row['goals_A'], row['goals_B'])
        kalman_system.update(row['team_B'], row['goals_B'], row['goals_A'])

    elo_A, elo_B, elo_diff = elo_system.get_ratings(team_A, team_B)
    ka_atk, ka_def, ka_unc_atk, ka_unc_def, \
    kb_atk, kb_def, kb_unc_atk, kb_unc_def = kalman_system.get_ratings(team_A, team_B)

    wins_A,  draws_A,  losses_A,  scored_A,  conc_A,  wscored_A,  wconc_A  = calculate_form_features(current_matches, team_A, FORM_N, elo_system.ratings)
    wins_B,  draws_B,  losses_B,  scored_B,  conc_B,  wscored_B,  wconc_B  = calculate_form_features(current_matches, team_B, FORM_N, elo_system.ratings)
    wins_A2, draws_A2, losses_A2, scored_A2, conc_A2, wscored_A2, wconc_A2 = calculate_form_features(current_matches, team_A, 2,      elo_system.ratings)
    wins_B2, draws_B2, losses_B2, scored_B2, conc_B2, wscored_B2, wconc_B2 = calculate_form_features(current_matches, team_B, 2,      elo_system.ratings)

    h2h_wins, h2h_draws, h2h_losses = calculate_h2h(current_matches, team_A, team_B)

    current_date = current_matches['date'].max() if len(current_matches) > 0 else pd.Timestamp.now()
    rest_A = calculate_days_rest(current_matches, team_A, current_date)
    rest_B = calculate_days_rest(current_matches, team_B, current_date)

    pr_features = compute_pagerank_features(current_matches)
    pr_A = get_pagerank(pr_features, team_A)
    pr_B = get_pagerank(pr_features, team_B)

    opp_elo_A, opp_scored_A, opp_conceded_A, n_opps_A = calculate_neighbourhood_features(
        current_matches, team_A, elo_system.ratings)
    opp_elo_B, opp_scored_B, opp_conceded_B, n_opps_B = calculate_neighbourhood_features(
        current_matches, team_B, elo_system.ratings)

    row = {
        'elo_A':        elo_A,
        'elo_B':        elo_B,
        'elo_diff':     elo_diff,
        'ka_atk':       ka_atk,   'ka_def':  ka_def,  'ka_unc_atk': ka_unc_atk, 'ka_unc_def': ka_unc_def,
        'kb_atk':       kb_atk,   'kb_def':  kb_def,  'kb_unc_atk': kb_unc_atk, 'kb_unc_def': kb_unc_def,
        'kalman_atk_diff': ka_atk - kb_atk,
        'kalman_def_diff': ka_def - kb_def,
        'wins_A':       wins_A,   'draws_A':  draws_A,  'losses_A': losses_A,
        'scored_A':     scored_A, 'conc_A':   conc_A,
        'wscored_A':    wscored_A,'wconc_A':  wconc_A,
        'wins_B':       wins_B,   'draws_B':  draws_B,  'losses_B': losses_B,
        'scored_B':     scored_B, 'conc_B':   conc_B,
        'wscored_B':    wscored_B,'wconc_B':  wconc_B,
        'wins_A2':      wins_A2,  'draws_A2': draws_A2, 'losses_A2': losses_A2,
        'scored_A2':    scored_A2,'conc_A2':  conc_A2,
        'wscored_A2':   wscored_A2,'wconc_A2': wconc_A2,
        'wins_B2':      wins_B2,  'draws_B2': draws_B2, 'losses_B2': losses_B2,
        'scored_B2':    scored_B2,'conc_B2':  conc_B2,
        'wscored_B2':   wscored_B2,'wconc_B2': wconc_B2,
        'h2h_wins':     h2h_wins, 'h2h_draws': h2h_draws, 'h2h_losses': h2h_losses,
        'rest_A':         rest_A,        'rest_B':         rest_B,
        'match_count_A':  int(((current_matches['team_A'] == team_A) | (current_matches['team_B'] == team_A)).sum()),
        'match_count_B':  int(((current_matches['team_A'] == team_B) | (current_matches['team_B'] == team_B)).sum()),
        'fifa_rank_A':    _FIFA_RANKINGS.get(team_A, _FIFA_DEFAULT),
        'fifa_rank_B':    _FIFA_RANKINGS.get(team_B, _FIFA_DEFAULT),
        'fifa_rank_diff': _FIFA_RANKINGS.get(team_A, _FIFA_DEFAULT) - _FIFA_RANKINGS.get(team_B, _FIFA_DEFAULT),
        'win_pr_A':       pr_A['win_pr'],  'win_pr_B':   pr_B['win_pr'],
        'win_pr_diff':    pr_A['win_pr']  - pr_B['win_pr'],
        'goal_pr_A':      pr_A['goal_pr'], 'goal_pr_B':  pr_B['goal_pr'],
        'goal_pr_diff':   pr_A['goal_pr'] - pr_B['goal_pr'],
        'hub_A':          pr_A['hub'],     'hub_B':      pr_B['hub'],
        'hub_diff':       pr_A['hub']     - pr_B['hub'],
        'auth_A':         pr_A['auth'],    'auth_B':     pr_B['auth'],
        'auth_diff':      pr_A['auth']    - pr_B['auth'],
        'opp_elo_A':      opp_elo_A,      'opp_elo_B':      opp_elo_B,
        'opp_elo_diff':   opp_elo_A - opp_elo_B,
        'opp_scored_A':   opp_scored_A,   'opp_scored_B':   opp_scored_B,
        'opp_conceded_A': opp_conceded_A, 'opp_conceded_B': opp_conceded_B,
        'n_opps_A':       n_opps_A,        'n_opps_B':       n_opps_B,
    }

    X = pd.DataFrame([row])[feature_cols]

    pred  = model.predict(X)[0]
    lam_A = max(float(pred[0]), 0.0)
    lam_B = max(float(pred[1]), 0.0)

    # ── Display score (amplified λ) + Poisson probability grid ───────────────
    display = predict_from_lambdas(lam_A, lam_B)
    grid    = score_grid(lam_A, lam_B)
    p_win_f, p_draw_f, p_loss_f = result_probabilities(grid)

    if p_win_f >= p_draw_f and p_win_f >= p_loss_f:
        outcome = 'win'
    elif p_draw_f >= p_win_f and p_draw_f >= p_loss_f:
        outcome = 'draw'
    else:
        outcome = 'loss'

    return {
        'team_A':     team_A,
        'team_B':     team_B,
        'goals_A':    display['goals_A'],
        'goals_B':    display['goals_B'],
        'outcome':    outcome,
        'p_win_A':    round(p_win_f  * 100, 1),
        'p_draw':     round(p_draw_f * 100, 1),
        'p_win_B':    round(p_loss_f * 100, 1),
        'prob_score': round(display['prob_score'] * 100, 1),
        'lam_A':      round(lam_A, 2),
        'lam_B':      round(lam_B, 2),
    }