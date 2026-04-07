# fetch_data.py

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)
TRAIN_PATH = "data/matches.csv"
TEST_PATH  = "data/wc2022.csv"


def fetch_data():
    print("Downloading international results dataset...")
    df = pd.read_csv(SOURCE_URL)

    # ── TRAIN: WC 2022 qualifiers + WC 2018 tournament (pre-WC2022 data) ──────
    wc2022_qual = df[
        (df['tournament'].str.contains('FIFA World Cup qualification', case=False, na=False)) &
        (df['date'] >= '2019-01-01') &
        (df['date'] <= '2022-11-19')   # strictly before WC 2022 kicks off
    ].copy()

    wc2018 = df[
        (df['tournament'] == 'FIFA World Cup') &
        (df['date'] >= '2018-06-14') &
        (df['date'] <= '2018-07-15')
    ].copy()

    train = pd.concat([wc2018, wc2022_qual], ignore_index=True)

    # ── TEST: WC 2022 tournament (64 games) ───────────────────────────────────
    wc2022 = df[
        (df['tournament'] == 'FIFA World Cup') &
        (df['date'] >= '2022-11-20') &
        (df['date'] <= '2022-12-18')
    ].copy()

    def clean(d):
        d = d.rename(columns={
            'home_team':  'team_A',
            'away_team':  'team_B',
            'home_score': 'goals_A',
            'away_score': 'goals_B',
        })
        d = d[['date', 'team_A', 'team_B', 'goals_A', 'goals_B']]
        d = d.dropna(subset=['goals_A', 'goals_B'])
        d['goals_A'] = d['goals_A'].astype(int)
        d['goals_B'] = d['goals_B'].astype(int)
        return d.sort_values('date').reset_index(drop=True)

    train  = clean(train)
    wc2022 = clean(wc2022)

    train.to_csv(TRAIN_PATH, index=False)
    wc2022.to_csv(TEST_PATH, index=False)

    print(f"Train: {len(train)} matches → {TRAIN_PATH}")
    print(f"  WC 2018 tournament : {len(wc2018)} games")
    print(f"  WC 2022 qualifiers : {len(wc2022_qual)} games")
    print(f"Test : {len(wc2022)} matches → {TEST_PATH}")


if __name__ == '__main__':
    fetch_data()
