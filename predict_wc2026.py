# predict_wc2026.py
# Predict all 2026 World Cup matches using the frozen model (walk-forward, no actuals).
# Exports predictions to Excel for review.

import pandas as pd

# Use Apr 2026 FIFA rankings for WC 2026 predictions
import features
features.set_fifa_rankings_year(2026)

from predict import predict_match

TRAIN_PATH = 'data/matches.csv'
TEST_PATH = 'data/wc2026.csv'
OUTPUT_PATH = 'predictions_wc2026_frozen.xlsx'

if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['date'])
    test_df = test_df.sort_values('date').reset_index(drop=True)

    # Walk-forward: update state with each predicted match
    history = train_df.copy()
    preds = []
    for _, match in test_df.iterrows():
        team_A = match['team_A']
        team_B = match['team_B']
        result = predict_match(team_A, team_B, history)
        preds.append({
            'date': match['date'],
            'team_A': team_A,
            'team_B': team_B,
            'pred_goals_A': result['goals_A'],
            'pred_goals_B': result['goals_B'],
            'p_win_A': result.get('p_win_A', None),
            'p_draw': result.get('p_draw', None),
            'p_win_B': result.get('p_win_B', None),
            'lam_A': result.get('lam_A', None),
            'lam_B': result.get('lam_B', None),
        })
        # Append predicted match to history for walk-forward
        new_row = pd.DataFrame([{
            'date': match['date'],
            'team_A': team_A,
            'team_B': team_B,
            'goals_A': result['goals_A'],
            'goals_B': result['goals_B'],
        }])
        history = pd.concat([history, new_row], ignore_index=True)
        history = history.sort_values('date').reset_index(drop=True)

    pred_df = pd.DataFrame(preds)
    pred_df.to_excel(OUTPUT_PATH, index=False)
    print(f"2026 predictions exported to {OUTPUT_PATH}")
