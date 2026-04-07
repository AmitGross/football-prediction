# features.py

import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx

ELO_START   = 1500
ELO_K       = 32
FORM_N      = 5

_FIFA_DEFAULT = 1200.0

# Versioned ranking files:
#   data/fifa_rankings_2022.csv  — Nov 2022 rankings (use with WC 2022 evaluation)
#   data/fifa_rankings_2026.csv  — Apr 2026 rankings (use with WC 2026 simulation)
_FIFA_RANKINGS_FILES = {
    2022: 'data/fifa_rankings_2022.csv',
    2026: 'data/fifa_rankings_2026.csv',
}
_FIFA_RANKINGS_DEFAULT_YEAR = 2026


def load_fifa_rankings(year=None):
    """
    Load FIFA rankings for the given year (2022 or 2026).
    Falls back to the default year (2026) if year is None or not found.
    Returns a dict {team_name: fifa_points}.
    """
    import os
    if year is None:
        year = _FIFA_RANKINGS_DEFAULT_YEAR
    csv_path = _FIFA_RANKINGS_FILES.get(year)
    if csv_path is None:
        print(f"[features] WARNING: No FIFA rankings file defined for year {year}. "
              f"Available: {list(_FIFA_RANKINGS_FILES.keys())}. Using default ({_FIFA_RANKINGS_DEFAULT_YEAR}).")
        csv_path = _FIFA_RANKINGS_FILES[_FIFA_RANKINGS_DEFAULT_YEAR]
    try:
        rankings = (
            pd.read_csv(csv_path)
            .set_index('team')['fifa_points']
            .to_dict()
        )
        print(f"[features] FIFA rankings loaded: {len(rankings)} teams "
              f"from {csv_path} (year={year})")
        return rankings
    except FileNotFoundError:
        print(f"[features] WARNING: FIFA rankings file not found: {csv_path}. "
              f"All teams will use default {_FIFA_DEFAULT} points.")
        return {}


# Module-level default — loaded once at import time (2026 for general use).
# Scripts that need a specific year should call set_fifa_rankings_year(year)
# before importing or building features.
_FIFA_RANKINGS = load_fifa_rankings(_FIFA_RANKINGS_DEFAULT_YEAR)


def set_fifa_rankings_year(year):
    """
    Switch the module-level FIFA rankings to the specified year.
    Call this at the top of a script before building any features.
    Example:
        import features
        features.set_fifa_rankings_year(2022)
    """
    global _FIFA_RANKINGS
    _FIFA_RANKINGS = load_fifa_rankings(year)


KALMAN_INIT_STRENGTH    = 0.0
KALMAN_INIT_UNCERTAINTY = 100.0
KALMAN_PROCESS_NOISE    = 5.0
KALMAN_MEASURE_NOISE    = 3.0


# ── 1. Elo ─────────────────────────────────────────────────────────────────────
class EloRating:
    def __init__(self):
        self.ratings = defaultdict(lambda: ELO_START)

    def get_ratings(self, team_A, team_B):
        elo_A    = self.ratings[team_A]
        elo_B    = self.ratings[team_B]
        elo_diff = elo_A - elo_B
        return elo_A, elo_B, elo_diff

    def update(self, team_A, team_B, goals_A, goals_B):
        expected_A = 1 / (1 + 10 ** ((self.ratings[team_B] - self.ratings[team_A]) / 400))
        expected_B = 1 - expected_A

        if goals_A > goals_B:
            actual_A, actual_B = 1.0, 0.0
        elif goals_A == goals_B:
            actual_A, actual_B = 0.5, 0.5
        else:
            actual_A, actual_B = 0.0, 1.0

        self.ratings[team_A] += ELO_K * (actual_A - expected_A)
        self.ratings[team_B] += ELO_K * (actual_B - expected_B)


# ── 2. Kalman filter — attack & defense split (walk-forward) ─────────────────
class KalmanRating:
    """
    Two independent 1-D Kalman filters per team:
      - attack:  tracks avg goals scored
      - defense: tracks avg goals conceded
    """
    def __init__(self):
        self.attack    = defaultdict(lambda: KALMAN_INIT_STRENGTH)
        self.defense   = defaultdict(lambda: KALMAN_INIT_STRENGTH)
        self.unc_atk   = defaultdict(lambda: KALMAN_INIT_UNCERTAINTY)
        self.unc_def   = defaultdict(lambda: KALMAN_INIT_UNCERTAINTY)

    def get_ratings(self, team_A, team_B):
        return (
            self.attack[team_A],  self.defense[team_A],
            self.unc_atk[team_A], self.unc_def[team_A],
            self.attack[team_B],  self.defense[team_B],
            self.unc_atk[team_B], self.unc_def[team_B],
        )

    def _update_one(self, state, unc, observation):
        P = unc + KALMAN_PROCESS_NOISE
        K = P / (P + KALMAN_MEASURE_NOISE)
        new_state = state + K * (observation - state)
        new_unc   = (1 - K) * P
        return new_state, new_unc

    def update(self, team, goals_for, goals_against):
        self.attack[team],  self.unc_atk[team] = self._update_one(
            self.attack[team],  self.unc_atk[team], goals_for)
        self.defense[team], self.unc_def[team] = self._update_one(
            self.defense[team], self.unc_def[team], goals_against)


# ── 3. Form features ───────────────────────────────────────────────────────────
def calculate_form_features(past_matches, team, N=FORM_N, elo_ratings=None):
    home = past_matches[past_matches['team_A'] == team].copy()
    home['gf'] = home['goals_A']
    home['ga'] = home['goals_B']
    home['opponent'] = home['team_B']

    away = past_matches[past_matches['team_B'] == team].copy()
    away['gf'] = away['goals_B']
    away['ga'] = away['goals_A']
    away['opponent'] = away['team_A']

    recent = pd.concat([home, away]).sort_values('date').tail(N)

    if len(recent) == 0:
        return 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    wins       = (recent['gf'] > recent['ga']).sum()
    draws      = (recent['gf'] == recent['ga']).sum()
    losses     = (recent['gf'] < recent['ga']).sum()
    avg_scored    = float(recent['gf'].mean())
    avg_conceded  = float(recent['ga'].mean())

    if elo_ratings is not None:
        weights = recent['opponent'].map(lambda opp: elo_ratings[opp])
        w_sum = float(weights.sum())
        if w_sum > 0:
            wscored = float((recent['gf'] * weights).sum() / w_sum)
            wconc   = float((recent['ga'] * weights).sum() / w_sum)
        else:
            wscored, wconc = avg_scored, avg_conceded
    else:
        wscored, wconc = avg_scored, avg_conceded

    return int(wins), int(draws), int(losses), avg_scored, avg_conceded, wscored, wconc


# ── 4. Head-to-head record ─────────────────────────────────────────────────────
def calculate_h2h(past_matches, team_A, team_B, N=5):
    h2h = past_matches[
        ((past_matches['team_A'] == team_A) & (past_matches['team_B'] == team_B)) |
        ((past_matches['team_A'] == team_B) & (past_matches['team_B'] == team_A))
    ].sort_values('date').tail(N)

    if len(h2h) == 0:
        return 0, 0, 0

    wins = draws = losses = 0
    for _, r in h2h.iterrows():
        if r['team_A'] == team_A:
            gf, ga = r['goals_A'], r['goals_B']
        else:
            gf, ga = r['goals_B'], r['goals_A']
        if gf > ga:   wins   += 1
        elif gf == ga: draws += 1
        else:          losses += 1

    return wins, draws, losses


# ── 5. Days rest ───────────────────────────────────────────────────────────────
def calculate_days_rest(past_matches, team, current_date):
    played = past_matches[
        (past_matches['team_A'] == team) | (past_matches['team_B'] == team)
    ]
    if len(played) == 0:
        return 30  # default: assume well rested
    last_date = played['date'].max()
    return max(0, (current_date - last_date).days)


# ── 6. Graph (undirected, each match = edge with goal attributes) ──────────────
def build_graph(past_matches):
    """
    nx.MultiGraph: teams are nodes, each match is an undirected edge.
    Edge attributes: goals_A, goals_B (the two values we want to predict).
    Multiple edges allowed — teams play each other many times.
    """
    G = nx.MultiGraph()
    for _, row in past_matches.iterrows():
        G.add_edge(
            row['team_A'],
            row['team_B'],
            goals_A = row['goals_A'],
            goals_B = row['goals_B']
        )
    return G


# ── 6b. PageRank + HITS graph features ────────────────────────────────────────
def compute_pagerank_features(matches, decay_lambda=0.5):
    """
    Two directed graphs with temporal decay on edge weights:
      win_graph  : winner → loser,  weight = goal margin × decay  (dominance chain)
      goal_graph : A → B,           weight = goals A scored vs B × decay  (offensive flow)

    Temporal decay: exp(-decay_lambda * years_ago) so recent matches matter more.
    decay_lambda=0.5 → half-weight at ~1.4 years ago.

    Computes:
      win_pr  — PageRank on win_graph  (overall dominance)
      goal_pr — PageRank on goal_graph (offensive flow)
      hub     — HITS hub on goal_graph  (scores goals against strong teams = attack quality)
      auth    — HITS authority on goal_graph (gets scored on by strong attackers = defensive vulnerability)
    """
    if len(matches) == 0:
        return {}

    ref_date = pd.to_datetime(matches['date']).max()

    win_graph  = nx.DiGraph()
    goal_graph = nx.DiGraph()

    for _, row in matches.iterrows():
        a, b   = row['team_A'], row['team_B']
        ga, gb = row['goals_A'], row['goals_B']

        days_ago = max(0, (ref_date - pd.to_datetime(row['date'])).days)
        decay    = float(np.exp(-decay_lambda * days_ago / 365.0))

        for src, dst, w in [(a, b, float(ga) * decay), (b, a, float(gb) * decay)]:
            if goal_graph.has_edge(src, dst):
                goal_graph[src][dst]['weight'] += w
            else:
                goal_graph.add_edge(src, dst, weight=w)

        if ga != gb:
            winner, loser, margin = (a, b, ga - gb) if ga > gb else (b, a, gb - ga)
            ew = float(margin) * decay
            if win_graph.has_edge(winner, loser):
                win_graph[winner][loser]['weight'] += ew
            else:
                win_graph.add_edge(winner, loser, weight=ew)

    if len(win_graph.nodes) < 2:
        return {}

    try:
        win_pr  = nx.pagerank(win_graph,  weight='weight', max_iter=300)
    except Exception:
        win_pr  = {n: 1.0 / len(win_graph) for n in win_graph.nodes}
    try:
        goal_pr = nx.pagerank(goal_graph, weight='weight', max_iter=300)
    except Exception:
        goal_pr = {n: 1.0 / len(goal_graph) for n in goal_graph.nodes}

    # HITS: hub = attack quality (scores against strong opponents)
    #       auth = defensive vulnerability (strong attackers score against you)
    try:
        hub_scores, auth_scores = nx.hits(goal_graph, max_iter=300, normalized=True)
    except Exception:
        n = max(len(goal_graph), 1)
        hub_scores  = {node: 1.0 / n for node in goal_graph.nodes}
        auth_scores = {node: 1.0 / n for node in goal_graph.nodes}

    all_teams = set(win_graph.nodes) | set(goal_graph.nodes)
    return {
        t: {
            'win_pr':  win_pr.get(t, 0.0),
            'goal_pr': goal_pr.get(t, 0.0),
            'hub':     hub_scores.get(t, 0.0),
            'auth':    auth_scores.get(t, 0.0),
        }
        for t in all_teams
    }


def get_pagerank(pr_features, team):
    return pr_features.get(team, {'win_pr': 0.0, 'goal_pr': 0.0, 'hub': 0.0, 'auth': 0.0})


# ── 6c. Walk-forward neighbourhood aggregation (1-hop message passing) ────────
def calculate_neighbourhood_features(past_matches, team, elo_ratings):
    """
    For each past opponent of `team`, look up:
      - their Elo rating  (schedule strength)
      - avg goals they scored in OTHER matches  (quality of defense team faced)
      - avg goals they conceded in OTHER matches (quality of offense team faced)

    This is exactly what a GNN round-1 message pass computes:
    aggregate neighbours' features into the focal node's representation.
    Fully walk-forward — only uses past_matches.
    """
    played = past_matches[
        (past_matches['team_A'] == team) | (past_matches['team_B'] == team)
    ]
    if len(played) == 0:
        return 0.0, 0.0, 0.0, 1500.0  # defaults

    opp_elos, opp_scored, opp_conceded = [], [], []

    for _, row in played.iterrows():
        opp = row['team_B'] if row['team_A'] == team else row['team_A']
        opp_elos.append(elo_ratings.get(opp, ELO_START))

        # opp's scoring record in matches NOT involving `team`
        opp_other = past_matches[
            ((past_matches['team_A'] == opp) | (past_matches['team_B'] == opp)) &
            (past_matches['team_A'] != team) & (past_matches['team_B'] != team)
        ]
        if len(opp_other) > 0:
            opp_gf = np.where(
                opp_other['team_A'] == opp,
                opp_other['goals_A'], opp_other['goals_B']
            ).mean()
            opp_ga = np.where(
                opp_other['team_A'] == opp,
                opp_other['goals_B'], opp_other['goals_A']
            ).mean()
            opp_scored.append(float(opp_gf))
            opp_conceded.append(float(opp_ga))

    avg_opp_elo      = float(np.mean(opp_elos)) if opp_elos else 1500.0
    avg_opp_scored   = float(np.mean(opp_scored))   if opp_scored   else 0.0
    avg_opp_conceded = float(np.mean(opp_conceded)) if opp_conceded else 0.0
    n_opponents      = len(set(
        list(played[played['team_A'] == team]['team_B']) +
        list(played[played['team_B'] == team]['team_A'])
    ))

    return avg_opp_elo, avg_opp_scored, avg_opp_conceded, float(n_opponents)


# ── 8. Main feature builder (walk-forward, no leakage) ────────────────────────
def build_features(matches, N=FORM_N):
    """
    Walk-forward: for match i, use only matches[0:i] to build features.
    Returns:
        X          — feature DataFrame (35 columns)
        y_goals_A  — Series of actual goals scored by team_A
        y_goals_B  — Series of actual goals scored by team_B
    """
    matches = matches.sort_values('date').reset_index(drop=True)

    # PageRank: computed from full dataset (structural signal, no target leakage).
    pr_features = compute_pagerank_features(matches)

    elo_system    = EloRating()
    kalman_system = KalmanRating()
    rows          = []
    ya            = []
    yb            = []

    for i, match in matches.iterrows():
        team_A  = match['team_A']
        team_B  = match['team_B']
        goals_A = match['goals_A']
        goals_B = match['goals_B']

        past = matches.iloc[:i]

        # Elo: GET first (no leakage), then UPDATE
        elo_A, elo_B, elo_diff = elo_system.get_ratings(team_A, team_B)
        elo_system.update(team_A, team_B, goals_A, goals_B)

        # Kalman: GET first (no leakage), then UPDATE both teams
        ka_atk, ka_def, ka_unc_atk, ka_unc_def, \
        kb_atk, kb_def, kb_unc_atk, kb_unc_def = kalman_system.get_ratings(team_A, team_B)
        kalman_system.update(team_A, goals_A, goals_B)
        kalman_system.update(team_B, goals_B, goals_A)

        # Form (home + away, past only) — last 5 and last 2, opponent-weighted
        wins_A,  draws_A,  losses_A,  scored_A,  conc_A,  wscored_A,  wconc_A  = calculate_form_features(past, team_A, N,   elo_system.ratings)
        wins_B,  draws_B,  losses_B,  scored_B,  conc_B,  wscored_B,  wconc_B  = calculate_form_features(past, team_B, N,   elo_system.ratings)
        wins_A2, draws_A2, losses_A2, scored_A2, conc_A2, wscored_A2, wconc_A2 = calculate_form_features(past, team_A, 2,   elo_system.ratings)
        wins_B2, draws_B2, losses_B2, scored_B2, conc_B2, wscored_B2, wconc_B2 = calculate_form_features(past, team_B, 2,   elo_system.ratings)

        # Head-to-head
        h2h_wins, h2h_draws, h2h_losses = calculate_h2h(past, team_A, team_B)

        # Days rest
        rest_A = calculate_days_rest(past, team_A, match['date'])
        rest_B = calculate_days_rest(past, team_B, match['date'])

        # Match count (walk-forward)
        match_count_A = int(((past['team_A'] == team_A) | (past['team_B'] == team_A)).sum())
        match_count_B = int(((past['team_A'] == team_B) | (past['team_B'] == team_B)).sum())

        # PageRank (pre-computed from full dataset)
        pr_A = get_pagerank(pr_features, team_A)
        pr_B = get_pagerank(pr_features, team_B)

        # Neighbourhood aggregation (walk-forward: 1-hop message passing)
        opp_elo_A, opp_scored_A, opp_conceded_A, n_opps_A = calculate_neighbourhood_features(
            past, team_A, elo_system.ratings)
        opp_elo_B, opp_scored_B, opp_conceded_B, n_opps_B = calculate_neighbourhood_features(
            past, team_B, elo_system.ratings)

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
            'match_count_A':  match_count_A, 'match_count_B':  match_count_B,
            'fifa_rank_A':    _FIFA_RANKINGS.get(team_A, _FIFA_DEFAULT),
            'fifa_rank_B':    _FIFA_RANKINGS.get(team_B, _FIFA_DEFAULT),
            'fifa_rank_diff': _FIFA_RANKINGS.get(team_A, _FIFA_DEFAULT) - _FIFA_RANKINGS.get(team_B, _FIFA_DEFAULT),
            # PageRank (global graph dominance, temporally decayed)
            'win_pr_A':       pr_A['win_pr'],  'win_pr_B':   pr_B['win_pr'],
            'win_pr_diff':    pr_A['win_pr']  - pr_B['win_pr'],
            'goal_pr_A':      pr_A['goal_pr'], 'goal_pr_B':  pr_B['goal_pr'],
            'goal_pr_diff':   pr_A['goal_pr'] - pr_B['goal_pr'],
            # HITS (hub = attack quality, auth = defensive vulnerability)
            'hub_A':          pr_A['hub'],     'hub_B':      pr_B['hub'],
            'hub_diff':       pr_A['hub']     - pr_B['hub'],
            'auth_A':         pr_A['auth'],    'auth_B':     pr_B['auth'],
            'auth_diff':      pr_A['auth']    - pr_B['auth'],
            # Neighbourhood / schedule-strength (walk-forward 1-hop aggregation)
            'opp_elo_A':      opp_elo_A,      'opp_elo_B':      opp_elo_B,
            'opp_elo_diff':   opp_elo_A - opp_elo_B,
            'opp_scored_A':   opp_scored_A,   'opp_scored_B':   opp_scored_B,
            'opp_conceded_A': opp_conceded_A, 'opp_conceded_B': opp_conceded_B,
            'n_opps_A':       n_opps_A,        'n_opps_B':       n_opps_B,
        }

        rows.append(row)
        ya.append(goals_A)
        yb.append(goals_B)

    X         = pd.DataFrame(rows)
    y_goals_A = pd.Series(ya, name='goals_A')
    y_goals_B = pd.Series(yb, name='goals_B')
    return X, y_goals_A, y_goals_B