import pandas as pd
import numpy as np

def knockout_stats(filepath, label):
    df = pd.read_excel(filepath)
    ko = df.tail(16).copy().reset_index(drop=True)
    rounds = (
        ['Round of 16'] * 8 +
        ['Quarter-Final'] * 4 +
        ['Semi-Final'] * 2 +
        ['3rd Place'] * 1 +
        ['Final'] * 1
    )
    ko.insert(0, 'Round', rounds)
    print(f'\n=== {label} ===')
    for _, r in ko.iterrows():
        tick = 'v' if r['Outcome_Correct'] else 'x'
        print(
            f"[{r['Round']:<14}] {r['team_A']:<20} vs {r['team_B']:<20}  "
            f"Pred {int(r['pred_goals_A'])}-{int(r['pred_goals_B'])}  "
            f"Actual {int(r['actual_goals_A'])}-{int(r['actual_goals_B'])}  "
            f"{tick}  RPS={r['RPS']:.3f}"
        )
    correct = int(ko['Outcome_Correct'].sum())
    n = len(ko)
    mae = (ko['MAE_A'].mean() + ko['MAE_B'].mean()) / 2
    rmse = np.sqrt(np.mean(
        list((ko['pred_goals_A'] - ko['actual_goals_A']) ** 2) +
        list((ko['pred_goals_B'] - ko['actual_goals_B']) ** 2)
    ))
    rps = ko['RPS'].mean()
    print(f'\n  Accuracy : {correct}/{n} ({correct/n*100:.1f}%)')
    print(f'  MAE      : {mae:.4f}')
    print(f'  RMSE     : {rmse:.4f}')
    print(f'  Mean RPS : {rps:.4f}')

knockout_stats('results_wc2022_retrain.xlsx',      'v1.0 Retrain — Knockout Stage')
knockout_stats('results_wc2022_retrain_v1.1.xlsx', 'v1.1 Retrain — Knockout Stage')
