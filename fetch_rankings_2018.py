"""
Fetch the official June 7, 2018 FIFA rankings (last ranking before WC 2018 started June 14).
Saves to data/fifa_rankings_2018.csv with columns: team, fifa_points
Source: tadhgfitzgerald/fifa_ranking on GitHub (covers 1993-present, real official data)
"""
import requests
import pandas as pd
from io import StringIO

NAME_MAP = {
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea',
    'USA': 'United States',
    'China PR': 'China',
    'Chinese Taipei': 'Taiwan',
    "Côte d'Ivoire": 'Ivory Coast',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Cape Verde Islands': 'Cape Verde',
    'Congo DR': 'DR Congo',
    'Congo': 'Republic of the Congo',
    'FYR Macedonia': 'North Macedonia',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Republic of Ireland': 'Republic of Ireland',
    'Northern Ireland': 'Northern Ireland',
}

url = 'https://raw.githubusercontent.com/tadhgfitzgerald/fifa_ranking/master/fifa_ranking.csv'
print(f"Downloading from {url}...")
r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
r.raise_for_status()

df = pd.read_csv(StringIO(r.text))
df['rank_date'] = pd.to_datetime(df['rank_date'])

snapshot = df[df['rank_date'] == '2018-06-07'].copy()
print(f"Teams in June 7 2018 snapshot: {len(snapshot)}")

snapshot['team'] = snapshot['country_full'].replace(NAME_MAP)
snapshot = snapshot[['team', 'total_points']].rename(columns={'total_points': 'fifa_points'})
snapshot = snapshot.sort_values('fifa_points', ascending=False).reset_index(drop=True)

out_path = 'data/fifa_rankings_2018.csv'
snapshot.to_csv(out_path, index=False)
print(f"Saved {len(snapshot)} teams to {out_path}")
print(snapshot.head(20).to_string(index=False))
