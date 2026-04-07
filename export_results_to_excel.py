import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# This script exports predictions, actuals, and loss metrics to Excel.
def export_results(preds, actuals, teams_A, teams_B, filename, rps_scores=None):
    df = pd.DataFrame({
        'team_A': teams_A,
        'team_B': teams_B,
        'pred_goals_A': preds[0],
        'pred_goals_B': preds[1],
        'actual_goals_A': actuals[0],
        'actual_goals_B': actuals[1],
    })
    df['MAE_A'] = np.abs(df['pred_goals_A'] - df['actual_goals_A'])
    df['MAE_B'] = np.abs(df['pred_goals_B'] - df['actual_goals_B'])
    df['RMSE_A'] = (df['pred_goals_A'] - df['actual_goals_A']) ** 2
    df['RMSE_B'] = (df['pred_goals_B'] - df['actual_goals_B']) ** 2
    if rps_scores is not None:
        df['RPS'] = rps_scores
    df.to_excel(filename, index=False)
    print(f'Excel file created: {filename}')

if __name__ == '__main__':
    # Example usage: load from npy or csv, or integrate into evaluate.py
    pass
