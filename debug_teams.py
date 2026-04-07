import pandas as pd

train = pd.read_csv('data/matches.csv')
test26 = pd.read_csv('data/wc2026.csv')
tt = set(train['team_A']) | set(train['team_B'])
t26 = set(test26['team_A']) | set(test26['team_B'])
m = t26 - tt
print(f'2026 coverage: {len(t26&tt)}/{len(t26)}')
print('Still missing:', sorted(m))
import sys; sys.exit(0)

# Mapping from 3-letter abbreviations (wc2026.csv) to full names (matches.csv)
ABBR_TO_FULL = {
    'ALG': 'Algeria',
    'ARG': 'Argentina',
    'AUS': 'Australia',
    'AUT': 'Austria',
    'BEL': 'Belgium',
    'BIH': 'Bosnia and Herzegovina',
    'BRA': 'Brazil',
    'CAN': 'Canada',
    'CIV': "Ivory Coast",
    'COD': 'DR Congo',
    'COL': 'Colombia',
    'CPV': 'Cape Verde',
    'CRO': 'Croatia',
    'CUW': 'Curacao',
    'CZE': 'Czech Republic',
    'ECU': 'Ecuador',
    'EGY': 'Egypt',
    'ENG': 'England',
    'ESP': 'Spain',
    'FRA': 'France',
    'GER': 'Germany',
    'GHA': 'Ghana',
    'HAI': 'Haiti',
    'IRN': 'Iran',
    'IRQ': 'Iraq',
    'JOR': 'Jordan',
    'JPN': 'Japan',
    'KOR': 'South Korea',
    'KSA': 'Saudi Arabia',
    'MAR': 'Morocco',
    'MEX': 'Mexico',
    'NED': 'Netherlands',
    'NOR': 'Norway',
    'NZL': 'New Zealand',
    'PAN': 'Panama',
    'PAR': 'Paraguay',
    'POR': 'Portugal',
    'QAT': 'Qatar',
    'RSA': 'South Africa',
    'SCO': 'Scotland',
    'SEN': 'Senegal',
    'SUI': 'Switzerland',
    'SWE': 'Sweden',
    'TUN': 'Tunisia',
    'TUR': 'Turkey',
    'URU': 'Uruguay',
    'USA': 'United States',
    'UZB': 'Uzbekistan',
}

df = pd.read_csv('data/wc2026.csv')
df['team_A'] = df['team_A'].map(ABBR_TO_FULL).fillna(df['team_A'])
df['team_B'] = df['team_B'].map(ABBR_TO_FULL).fillna(df['team_B'])
df.to_csv('data/wc2026.csv', index=False)
print("wc2026.csv updated with full team names.")
print(df[['team_A', 'team_B']].head(10))
