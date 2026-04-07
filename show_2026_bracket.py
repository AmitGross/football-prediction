import pandas as pd

df = pd.read_excel('predictions_wc2026_full.xlsx')
print(df.columns.tolist())
print(df.head(5))
print(f"\nTotal rows: {len(df)}")
print("\nUnique stages:")
if 'stage' in df.columns:
    print(df['stage'].unique())
elif 'Stage' in df.columns:
    print(df['Stage'].unique())
print("\nFull data:")
print(df.to_string())
