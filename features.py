# features.py

import pandas as pd
import numpy as np
from collections import defaultdict

ELO_START   = 1500
ELO_K       = 32
FORM_N      = 5

_FIFA_DEFAULT = 1200.0

# Versioned ranking files — one snapshot per WC year, just before the tournament:
#   data/fifa_rankings_2018.csv  — Jun 2018 rankings (use with WC 2018 evaluation)
#   data/fifa_rankings_2022.csv  — Nov 2022 rankings (use with WC 2022 evaluation)
#   data/fifa_rankings_2026.csv  — Apr 2026 rankings (use with WC 2026 simulation)
_FIFA_RANKINGS_FILES = {
    2018: 'data/fifa_rankings_2018.csv',
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


# ── 5b. Goal-difference std (volatility) ──────────────────────────────────────
def calculate_goal_diff_std(past_matches, team, N=5):
    """
    Standard deviation of goal difference (GF-GA) in the team's last N matches.
    Returns 0.0 if fewer than 2 past matches exist (std undefined).
    """
    home = past_matches[past_matches['team_A'] == team][['date', 'goals_A', 'goals_B']].copy()
    home['gd'] = home['goals_A'] - home['goals_B']

    away = past_matches[past_matches['team_B'] == team][['date', 'goals_A', 'goals_B']].copy()
    away['gd'] = away['goals_B'] - away['goals_A']

    recent = pd.concat([home[['date', 'gd']], away[['date', 'gd']]]).sort_values('date').tail(N)
    if len(recent) < 2:
        return 0.0
    return float(recent['gd'].std())


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


# ── 6c. Walk-forward neighbourhood aggregation (1-hop message passing) ────────
def calculate_neighbourhood_features(past_matches, team, elo_ratings, top_pct=0.70):
    """
    For each past opponent of `team`, aggregates:
      - their Elo rating  (schedule strength)
      - avg goals they scored/conceded in OTHER matches
      - performance-aware features: outcomes and goal diffs vs those opponents

    This is exactly what a GNN round-1 message pass computes:
    aggregate neighbours' features into the focal node's representation.
    Fully walk-forward — only uses past_matches.

    Returns a dict of feature values (both original and new).
    top_pct: percentile threshold for "top teams" (default 0.70 → top 30%).
    """
    _defaults = {
        'avg_opp_elo':               1500.0,
        'avg_opp_scored':            0.0,
        'avg_opp_conceded':          0.0,
        'n_opponents':               0.0,
        # --- NEW FEATURES START ---
        'weighted_opp_elo':          0.0,   # outcome-weighted opp Elo (+1 win / 0 draw / -1 loss)
        'win_rate_vs_top_teams':     0.0,   # win rate against opponents above top_pct Elo threshold
        'avg_goal_diff_vs_opp':      0.0,   # average GF-GA across all matches vs opponents
        'weighted_goal_diff_by_opp': 0.0,   # goal diff scaled by opponent Elo strength
        # --- NEW FEATURES END ---
    }

    played = past_matches[
        (past_matches['team_A'] == team) | (past_matches['team_B'] == team)
    ]
    if len(played) == 0:
        return _defaults

    # Elo top-team threshold from current ratings (top 30% = above the 70th percentile)
    all_elos = list(elo_ratings.values()) if elo_ratings else [ELO_START]
    top_elo_threshold = float(np.percentile(all_elos, top_pct * 100))

    opp_elos, opp_scored, opp_conceded = [], [], []
    # --- NEW FEATURES START ---
    outcome_weights = []  # list of (opp_elo, outcome) where outcome ∈ {+1, 0, -1}
    goal_diffs      = []  # list of (goal_diff, opp_elo) for team in each match
    # --- NEW FEATURES END ---

    for _, row in played.iterrows():
        is_home = row['team_A'] == team
        opp     = row['team_B'] if is_home else row['team_A']
        opp_elo = elo_ratings.get(opp, ELO_START)
        opp_elos.append(opp_elo)

        # Goals for/against team in this specific match
        gf = float(row['goals_A'] if is_home else row['goals_B'])
        ga = float(row['goals_B'] if is_home else row['goals_A'])

        # --- NEW FEATURES START ---
        # Outcome encoding: +1 win, 0 draw, -1 loss
        if gf > ga:
            outcome = 1.0
        elif gf == ga:
            outcome = 0.0
        else:
            outcome = -1.0
        outcome_weights.append((opp_elo, outcome))
        goal_diffs.append((gf - ga, opp_elo))   # raw goal difference and opponent Elo
        # --- NEW FEATURES END ---

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

    avg_opp_elo      = float(np.mean(opp_elos))    if opp_elos    else 1500.0
    avg_opp_scored   = float(np.mean(opp_scored))   if opp_scored   else 0.0
    avg_opp_conceded = float(np.mean(opp_conceded)) if opp_conceded else 0.0
    n_opponents      = float(len(set(
        list(played[played['team_A'] == team]['team_B']) +
        list(played[played['team_B'] == team]['team_A'])
    )))

    # --- NEW FEATURES START ---

    # 1. Outcome-weighted opponent Elo: mean(outcome * opp_elo)
    #    High value → won against strong opponents; low → lost against strong opponents
    weighted_opp_elo = float(np.mean([elo * out for elo, out in outcome_weights]))

    # 2. Win rate vs top teams (opponents at or above the top_pct Elo percentile)
    top_matches = [(elo, out) for elo, out in outcome_weights if elo >= top_elo_threshold]
    if top_matches:
        win_rate_vs_top_teams = float(np.mean([1.0 if out > 0 else 0.0 for _, out in top_matches]))
    else:
        win_rate_vs_top_teams = 0.0  # no top-team encounters yet

    # 3. Average goal difference (GF-GA) across all matches vs opponents
    avg_goal_diff_vs_opp = float(np.mean([gd for gd, _ in goal_diffs])) if goal_diffs else 0.0

    # 4. Goal difference weighted by opponent Elo strength (relative to avg opponent quality)
    #    A +2 GD vs Elo-1800 opponent counts more than +2 vs Elo-1400
    if goal_diffs and avg_opp_elo > 0:
        weighted_goal_diff_by_opp = float(
            np.mean([(gd * elo) / avg_opp_elo for gd, elo in goal_diffs])
        )
    else:
        weighted_goal_diff_by_opp = 0.0

    # --- NEW FEATURES END ---

    return {
        'avg_opp_elo':               avg_opp_elo,
        'avg_opp_scored':            avg_opp_scored,
        'avg_opp_conceded':          avg_opp_conceded,
        'n_opponents':               n_opponents,
        # --- NEW FEATURES START ---
        'weighted_opp_elo':          weighted_opp_elo,
        'win_rate_vs_top_teams':     win_rate_vs_top_teams,
        'avg_goal_diff_vs_opp':      avg_goal_diff_vs_opp,
        'weighted_goal_diff_by_opp': weighted_goal_diff_by_opp,
        # --- NEW FEATURES END ---
    }


# ── 8. Main feature builder (walk-forward, no leakage) ────────────────────────
def build_features(matches, N=FORM_N):
    """
    Walk-forward: for match i, use only matches[0:i] to build features.
    Returns:
        X          — feature DataFrame (68 columns in v1.6)
        y_goals_A  — Series of actual goals scored by team_A
        y_goals_B  — Series of actual goals scored by team_B
    """
    matches = matches.sort_values('date').reset_index(drop=True)

    # Stage feature support: if 'round' column is present, map it to numeric features.
    # For training data (matches.csv), no round column → all stage features default to 0.
    _ROUND_TO_STAGE = {
        'group':    (0, 1),   # (is_knockout, round_number)
        'r32':      (1, 2),
        'r16':      (1, 2),
        'qf':       (1, 3),
        'sf':       (1, 4),
        '3rd':      (1, 4),
        'final':    (1, 5),
        'f':        (1, 5),
    }
    has_round_col = 'round' in matches.columns

    elo_system    = EloRating()
    rows          = []
    ya            = []
    yb            = []
    games_played_in_tournament = {}   # team → count of tournament games played so far

    for i, match in matches.iterrows():
        team_A  = match['team_A']
        team_B  = match['team_B']
        goals_A = match['goals_A']
        goals_B = match['goals_B']

        past = matches.iloc[:i]

        # Elo: GET first (no leakage), then UPDATE
        elo_A, elo_B, elo_diff = elo_system.get_ratings(team_A, team_B)
        elo_system.update(team_A, team_B, goals_A, goals_B)

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

        # Neighbourhood aggregation (walk-forward: 1-hop message passing)
        nbr_A = calculate_neighbourhood_features(past, team_A, elo_system.ratings)
        nbr_B = calculate_neighbourhood_features(past, team_B, elo_system.ratings)

        # Goal-difference std (volatility, last 5 matches) — v1.6
        goal_diff_std_A = calculate_goal_diff_std(past, team_A)
        goal_diff_std_B = calculate_goal_diff_std(past, team_B)

        # Stage features — v1.6
        # If round column present, map to (is_knockout, round_number); else default 0.
        if has_round_col and pd.notna(match.get('round', None)):
            rnd_key = str(match['round']).strip().lower()
            is_knockout, round_number = _ROUND_TO_STAGE.get(rnd_key, (0, 0))
        else:
            is_knockout, round_number = 0, 0
        games_in_tournament_A = games_played_in_tournament.get(team_A, 0)
        games_in_tournament_B = games_played_in_tournament.get(team_B, 0)

        row = {
            'elo_A':        elo_A,
            'elo_B':        elo_B,
            'elo_diff':     elo_diff,
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
            # Neighbourhood / schedule-strength (walk-forward 1-hop aggregation)
            'opp_elo_A':      nbr_A['avg_opp_elo'],        'opp_elo_B':      nbr_B['avg_opp_elo'],
            'opp_elo_diff':   nbr_A['avg_opp_elo']       - nbr_B['avg_opp_elo'],
            'opp_scored_A':   nbr_A['avg_opp_scored'],    'opp_scored_B':   nbr_B['avg_opp_scored'],
            'opp_conceded_A': nbr_A['avg_opp_conceded'],  'opp_conceded_B': nbr_B['avg_opp_conceded'],
            'n_opps_A':       nbr_A['n_opponents'],         'n_opps_B':       nbr_B['n_opponents'],
            # --- NEW FEATURES START ---
            # Outcome-weighted opponent Elo: mean(outcome * opp_elo), +1 win / 0 draw / -1 loss
            'weighted_opp_elo_A':        nbr_A['weighted_opp_elo'],
            'weighted_opp_elo_B':        nbr_B['weighted_opp_elo'],
            'weighted_opp_elo_diff':     nbr_A['weighted_opp_elo']          - nbr_B['weighted_opp_elo'],
            # Win rate vs top-30% Elo opponents
            'win_rate_vs_top_A':         nbr_A['win_rate_vs_top_teams'],
            'win_rate_vs_top_B':         nbr_B['win_rate_vs_top_teams'],
            'win_rate_vs_top_diff':      nbr_A['win_rate_vs_top_teams']     - nbr_B['win_rate_vs_top_teams'],
            # Average goal difference (GF-GA) across matches vs past opponents
            'avg_goal_diff_vs_opp_A':    nbr_A['avg_goal_diff_vs_opp'],
            'avg_goal_diff_vs_opp_B':    nbr_B['avg_goal_diff_vs_opp'],
            'avg_goal_diff_vs_opp_diff': nbr_A['avg_goal_diff_vs_opp']     - nbr_B['avg_goal_diff_vs_opp'],
            # Strength-weighted goal difference (GD × opp_elo / avg_opp_elo)
            'wtd_goal_diff_opp_A':       nbr_A['weighted_goal_diff_by_opp'],
            'wtd_goal_diff_opp_B':       nbr_B['weighted_goal_diff_by_opp'],
            'wtd_goal_diff_opp_diff':    nbr_A['weighted_goal_diff_by_opp'] - nbr_B['weighted_goal_diff_by_opp'],
            # --- NEW FEATURES END ---
            # v1.6 — Stage context + volatility
            'is_knockout':                    is_knockout,
            'round_number':                   round_number,
            'games_in_tournament_A':          games_in_tournament_A,
            'games_in_tournament_B':          games_in_tournament_B,
            'goal_diff_std_A':                goal_diff_std_A,
            'goal_diff_std_B':                goal_diff_std_B,
        }

        rows.append(row)
        ya.append(goals_A)
        yb.append(goals_B)

        # Increment tournament-game counters only for round-tagged rows
        if has_round_col and pd.notna(match.get('round', None)):
            games_played_in_tournament[team_A] = games_in_tournament_A + 1
            games_played_in_tournament[team_B] = games_in_tournament_B + 1

    X         = pd.DataFrame(rows)
    y_goals_A = pd.Series(ya, name='goals_A')
    y_goals_B = pd.Series(yb, name='goals_B')
    return X, y_goals_A, y_goals_B