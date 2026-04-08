import pandas as pd, numpy as np

for fname in ['results_wc2022_frozen.xlsx', 'results_wc2022_retrain.xlsx']:
    df = pd.read_excel(fname)
    print('\\n=== ' + fname + ' ===')
    print(df.columns.tolist())
    n = len(df)
    mae = (df['MAE_A'].mean() + df['MAE_B'].mean()) / 2
    rmse = __import__('numpy').sqrt(((df['RMSE_A']**2 + df['RMSE_B']**2) / 2).mean())
    acc = df['Outcome_Correct'].mean() * 100
    rps = df['RPS'].mean()
    print('Matches: ' + str(n))
    print('MAE: ' + str(round(mae, 4)))
    print('RMSE combined: ' + str(round(rmse, 4)))
    print('Outcome accuracy: ' + str(round(acc, 1)) + '%')
    print('Mean RPS: ' + str(round(rps, 4)))
