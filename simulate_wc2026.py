# simulate_wc2026.py
# Full 2026 World Cup tournament simulation (frozen model, no actuals).
# Group stage → standings → R32 → R16 → QF → SF → 3rd place → Final
# Exports all results to predictions_wc2026_full.xlsx

import pandas as pd
import numpy as np

# Use Apr 2026 FIFA rankings for WC 2026 simulation
import features
features.set_fifa_rankings_year(2026)

from predict import predict_match

MODEL_VERSION = 'v1.1'   # bump when features or model architecture changes

TRAIN_PATH = 'data/matches.csv'
GROUP_PATH = 'data/wc2026.csv'
OUTPUT_PATH = f'predictions_wc2026_full_{MODEL_VERSION}.xlsx'

# ── Group definitions (order matches wc2026.csv) ──────────────────────────────
GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# R32 bracket: (groupX_rank, groupY_rank) pairs
# Official 2026 bracket pairing (simplified):
R32_BRACKET = [
    ('A', 1, 'B', 2), ('B', 1, 'A', 2),
    ('C', 1, 'D', 2), ('D', 1, 'C', 2),
    ('E', 1, 'F', 2), ('F', 1, 'E', 2),
    ('G', 1, 'H', 2), ('H', 1, 'G', 2),
    ('I', 1, 'J', 2), ('J', 1, 'I', 2),
    ('K', 1, 'L', 2), ('L', 1, 'K', 2),
    # 8 third-place spots: best 8 of 12 groups, paired against remaining 1st seeds
    # Simplified: pair 3rd-place teams as opponents in last 8 R32 slots
]


def predict(team_A, team_B, history):
    result = predict_match(team_A, team_B, history)
    new_row = pd.DataFrame([{
        'date': pd.Timestamp('2026-07-01'),
        'team_A': team_A,
        'team_B': team_B,
        'goals_A': result['goals_A'],
        'goals_B': result['goals_B'],
    }])
    history = pd.concat([history, new_row], ignore_index=True)
    return result, history


def simulate_group_stage(history):
    group_df = pd.read_csv(GROUP_PATH, parse_dates=['date'])
    group_df = group_df.sort_values('date').reset_index(drop=True)

    match_results = []
    for _, match in group_df.iterrows():
        team_A, team_B = match['team_A'], match['team_B']
        result, history = predict(team_A, team_B, history)
        match_results.append({
            'stage': 'Group',
            'date': match['date'],
            'team_A': team_A,
            'team_B': team_B,
            'pred_goals_A': result['goals_A'],
            'pred_goals_B': result['goals_B'],
            'p_win_A': result.get('p_win_A'),
            'p_draw': result.get('p_draw'),
            'p_win_B': result.get('p_win_B'),
        })

    return match_results, history


def compute_standings(match_results):
    """Compute group table from predicted results."""
    standings = {g: {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
                 for g, teams in GROUPS.items()}

    # Map each team to its group
    team_to_group = {t: g for g, teams in GROUPS.items() for t in teams}

    for m in match_results:
        if m['stage'] != 'Group':
            continue
        a, b = m['team_A'], m['team_B']
        ga, gb = m['pred_goals_A'], m['pred_goals_B']
        g = team_to_group.get(a)
        if not g:
            continue

        standings[g][a]['gf'] += ga
        standings[g][a]['gd'] += ga - gb
        standings[g][b]['gf'] += gb
        standings[g][b]['gd'] += gb - ga

        if ga > gb:
            standings[g][a]['pts'] += 3
        elif ga == gb:
            standings[g][a]['pts'] += 1
            standings[g][b]['pts'] += 1
        else:
            standings[g][b]['pts'] += 3

    # Sort each group
    sorted_groups = {}
    for g, table in standings.items():
        ranked = sorted(table.items(), key=lambda x: (-x[1]['pts'], -x[1]['gd'], -x[1]['gf']))
        sorted_groups[g] = [(team, stats, rank+1) for rank, (team, stats) in enumerate(ranked)]

    return sorted_groups


def get_qualifiers(sorted_groups):
    """Return 1st, 2nd per group + best 8 third-place teams."""
    firsts  = {g: sorted_groups[g][0] for g in sorted_groups}
    seconds = {g: sorted_groups[g][1] for g in sorted_groups}
    thirds  = [sorted_groups[g][2] for g in sorted_groups]

    # Best 8 thirds by pts, gd, gf
    thirds_sorted = sorted(thirds, key=lambda x: (-x[1]['pts'], -x[1]['gd'], -x[1]['gf']))
    best8_thirds = thirds_sorted[:8]

    return firsts, seconds, best8_thirds


def simulate_knockout_round(matches, stage, history, date_str):
    """Run a list of (teamA, teamB) knockout matches. Returns winners and match records."""
    results = []
    winners = []
    for team_A, team_B in matches:
        result, history = predict(team_A, team_B, history)
        ga, gb = result['goals_A'], result['goals_B']
        # In knockouts always have a winner — use probabilities if draw predicted
        if ga > gb:
            winner = team_A
        elif gb > ga:
            winner = team_B
        else:
            # Penalty tiebreak: higher p_win_A wins
            winner = team_A if result.get('p_win_A', 50) >= result.get('p_win_B', 50) else team_B
        winners.append(winner)
        results.append({
            'stage': stage,
            'date': date_str,
            'team_A': team_A,
            'team_B': team_B,
            'pred_goals_A': ga,
            'pred_goals_B': gb,
            'winner': winner,
            'p_win_A': result.get('p_win_A'),
            'p_draw': result.get('p_draw'),
            'p_win_B': result.get('p_win_B'),
        })
    return results, winners, history


if __name__ == '__main__':
    print("Loading training data...")
    history = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
    history = history.sort_values('date').reset_index(drop=True)

    all_results = []

    # ── Group Stage ───────────────────────────────────────────────────────────
    print("\n=== GROUP STAGE ===")
    group_results, history = simulate_group_stage(history)
    all_results.extend(group_results)
    for r in group_results:
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']}")

    # ── Standings ─────────────────────────────────────────────────────────────
    sorted_groups = compute_standings(all_results)
    firsts, seconds, best8_thirds = get_qualifiers(sorted_groups)

    print("\n=== GROUP STANDINGS ===")
    for g in sorted(sorted_groups):
        print(f"  Group {g}:")
        for team, stats, rank in sorted_groups[g]:
            print(f"    {rank}. {team:<30} Pts:{stats['pts']}  GD:{stats['gd']:+d}  GF:{stats['gf']}")

    print("\n=== QUALIFIED TEAMS ===")
    for g in sorted(firsts): print(f"  1st Group {g}: {firsts[g][0]}")
    for g in sorted(seconds): print(f"  2nd Group {g}: {seconds[g][0]}")
    print("  Best 8 3rd-place teams:")
    for t, stats, _ in best8_thirds: print(f"    {t}")

    # ── Round of 32 ───────────────────────────────────────────────────────────
    print("\n=== ROUND OF 32 ===")
    r32_matches = []
    for (gA, r1, gB, r2) in R32_BRACKET:
        t1 = firsts[gA][0] if r1 == 1 else seconds[gA][0]
        t2 = firsts[gB][0] if r2 == 1 else seconds[gB][0]
        r32_matches.append((t1, t2))
    # Pair best 8 thirds vs remaining (use first from groups I-L and reversed)
    third_teams = [t[0] for t in best8_thirds]
    remaining = [('3rd', t) for t in third_teams[:4]]
    # Pair: each third vs corresponding 1st place from opposite half
    extra_pairs = list(zip(third_teams[:4], third_teams[4:]))
    r32_matches.extend(extra_pairs)

    r32_results, r32_winners, history = simulate_knockout_round(r32_matches, 'R32', history, '2026-07-04')
    all_results.extend(r32_results)
    for r in r32_results:
        tick = '→'
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']:<25}  {tick} {r['winner']}")

    # ── Round of 16 ───────────────────────────────────────────────────────────
    print("\n=== ROUND OF 16 ===")
    r16_matches = [(r32_winners[i], r32_winners[i+1]) for i in range(0, len(r32_winners), 2)]
    r16_results, r16_winners, history = simulate_knockout_round(r16_matches, 'R16', history, '2026-07-08')
    all_results.extend(r16_results)
    for r in r16_results:
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']:<25}  → {r['winner']}")

    # ── Quarter-Finals ────────────────────────────────────────────────────────
    print("\n=== QUARTER-FINALS ===")
    qf_matches = [(r16_winners[i], r16_winners[i+1]) for i in range(0, len(r16_winners), 2)]
    qf_results, qf_winners, history = simulate_knockout_round(qf_matches, 'QF', history, '2026-07-11')
    all_results.extend(qf_results)
    for r in qf_results:
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']:<25}  → {r['winner']}")

    # ── Semi-Finals ───────────────────────────────────────────────────────────
    print("\n=== SEMI-FINALS ===")
    sf_matches = [(qf_winners[i], qf_winners[i+1]) for i in range(0, len(qf_winners), 2)]
    sf_results, sf_winners, history = simulate_knockout_round(sf_matches, 'SF', history, '2026-07-14')
    all_results.extend(sf_results)
    sf_losers = []
    for idx, r in enumerate(sf_results):
        loser = r['team_B'] if r['winner'] == r['team_A'] else r['team_A']
        sf_losers.append(loser)
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']:<25}  → {r['winner']}")

    # ── 3rd Place ─────────────────────────────────────────────────────────────
    print("\n=== 3RD PLACE MATCH ===")
    third_results, third_winners, history = simulate_knockout_round(
        [(sf_losers[0], sf_losers[1])], '3rd Place', history, '2026-07-18')
    all_results.extend(third_results)
    for r in third_results:
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']:<25}  → {r['winner']}")

    # ── Final ─────────────────────────────────────────────────────────────────
    print("\n=== FINAL ===")
    final_results, final_winners, history = simulate_knockout_round(
        [(sf_winners[0], sf_winners[1])], 'Final', history, '2026-07-19')
    all_results.extend(final_results)
    for r in final_results:
        print(f"  {r['team_A']:<25} {r['pred_goals_A']}-{r['pred_goals_B']}  {r['team_B']:<25}  → {r['winner']}")
    print(f"\n  🏆 PREDICTED CHAMPION: {final_winners[0]}")

    # ── Export to Excel ───────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        # All matches
        pd.DataFrame(all_results).to_excel(writer, sheet_name='All Matches', index=False)

        # Group standings
        rows = []
        for g in sorted(sorted_groups):
            for team, stats, rank in sorted_groups[g]:
                rows.append({'Group': g, 'Rank': rank, 'Team': team,
                             'Points': stats['pts'], 'GD': stats['gd'], 'GF': stats['gf'],
                             'Qualified': rank <= 2})
        pd.DataFrame(rows).to_excel(writer, sheet_name='Group Standings', index=False)

        # Summary
        summary = [{'Stage': r['stage'], 'Team A': r['team_A'], 'Score': f"{r['pred_goals_A']}-{r['pred_goals_B']}",
                    'Team B': r['team_B'], 'Winner': r.get('winner', ''),
                    'p_win_A': r.get('p_win_A'), 'p_draw': r.get('p_draw'), 'p_win_B': r.get('p_win_B')}
                   for r in all_results if r['stage'] != 'Group']
        pd.DataFrame(summary).to_excel(writer, sheet_name='Knockouts', index=False)

    print(f"\nFull tournament predictions exported to {OUTPUT_PATH}")
