# batch_predict.py
# Batch prediction for all matches in a test file (e.g., wc2026.csv) using the frozen model.

import pandas as pd
from predict import predict_match

TRAIN_PATH = 'data/matches.csv'
TEST_PATH = 'data/wc2022.csv'
OUTPUT_PATH = 'predictions_wc2022_frozen.csv'


if __name__ == '__main__':
    # Load training data (for Elo/Kalman state)
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
    train_df = train_df.sort_values('date').reset_index(drop=True)

    # Load test matches
    test_df = pd.read_csv(TEST_PATH, parse_dates=['date'])
    test_df = test_df.sort_values('date').reset_index(drop=True)

    # Start with all training matches
    state_df = train_df.copy()
    preds = []
    for _, match in test_df.iterrows():
        team_A = match['team_A']
        team_B = match['team_B']
        # Predict using all matches up to this point
        result = predict_match(team_A, team_B, state_df)
        preds.append(result)
        # Append the predicted match to state for walk-forward
        # Use predicted goals for both teams
        new_row = {
            'date': match['date'],
            'team_A': team_A,
            'team_B': team_B,
            'goals_A': result['goals_A'],
            'goals_B': result['goals_B']
        }
        state_df = pd.concat([state_df, pd.DataFrame([new_row])], ignore_index=True)
        state_df = state_df.sort_values('date').reset_index(drop=True)

    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")
