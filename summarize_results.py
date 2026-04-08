import pandas as pd, numpy as np

for fname in ['results_wc2022_frozen.xlsx', 'results_wc2022_retrain.xlsx']:
    df = pd.read_excel(fname)
    print(f'\n=== {fname} ===')
    print(df.columns.tolist())
    n = len(df)
    mae = (df['MAE_A'].mean() + df['MAE_B'].mean()) / 2
    rmse = np.sqrt(((df['RMSE_A']**2 + df['RMSE_B']**2) / 2).mean())
    acc = df['Outcome_Correct'].mean() * 100
    rps = df['RPS'].mean()
    print(f'Matches: {n}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE combined: {rmse:.4f}')
    print(f'Outcome accuracy: {acc:.1f}%')
    print(f'Mean RPS: {rps:.4f}')
