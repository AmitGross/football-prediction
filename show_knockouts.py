import pandas as pd
import numpy as np

df = pd.read_excel('results_wc2022_retrain.xlsx')

# Knockout stage = last 16 matches (R16: 8, QF: 4, SF: 2, 3rd: 1, Final: 1)
ko = df.tail(16).copy().reset_index(drop=True)

rounds = (
    ['Round of 16'] * 8 +
    ['Quarter-Final'] * 4 +
    ['Semi-Final'] * 2 +
    ['3rd Place'] * 1 +
    ['Final'] * 1
)
ko.insert(0, 'Round', rounds)

print("\n=== WC 2022 KNOCKOUT STAGE — RETRAIN MODEL ===\n")
for _, r in ko.iterrows():
    tick = 'v' if r['Outcome_Correct'] else 'x'
    print(f"[{r['Round']:<14}] {r['team_A']:<20} vs {r['team_B']:<20}  "
          f"Pred {int(r['pred_goals_A'])}-{int(r['pred_goals_B'])}  "
          f"Actual {int(r['actual_goals_A'])}-{int(r['actual_goals_B'])}  "
          f"{tick}  RPS={r['RPS']:.3f}")

correct = ko['Outcome_Correct'].sum()
n = len(ko)
acc = correct / n * 100
mae = (ko['MAE_A'].mean() + ko['MAE_B'].mean()) / 2
rmse = np.sqrt(np.mean(
    list((ko['pred_goals_A'] - ko['actual_goals_A'])**2) +
    list((ko['pred_goals_B'] - ko['actual_goals_B'])**2)
))
rps = ko['RPS'].mean()

print(f"\n--- Knockout Stage Summary ---")
print(f"Matches     : {n}")
print(f"Correct     : {correct}/{n}  ({acc:.1f}%)")
print(f"MAE         : {mae:.3f}")
print(f"RMSE        : {rmse:.3f}")
print(f"Mean RPS    : {rps:.4f}")
