# fetch_data.py
#
# Fetch training data for a given World Cup year.
# The training window is always the 4 years of competitive matches leading up
# to that tournament (previous WC tournament + qualification cycle).
#
#   python fetch_data.py --year 2022   → WC 2018 tournament + WC 2022 qualifiers
#   python fetch_data.py --year 2026   → WC 2022 tournament + WC 2026 qualifiers

import argparse
import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)
TRAIN_PATH = "data/matches.csv"

# ── Per-year config ───────────────────────────────────────────────────────────
# Each entry describes:
#   prev_wc   : (date_from, date_to)  — the previous WC tournament matches
#   qual      : (date_from, date_to)  — the qualification cycle for THIS wc
#   test      : (date_from, date_to)  — THIS wc tournament (used as eval set)
#   test_path : where to save the eval set

WC_CONFIG = {
    2018: {
        'prev_wc': ('2014-06-12', '2014-07-13'),   # WC 2014 tournament
        'qual':    ('2015-01-01', '2018-06-13'),   # strictly before WC 2018 kicks off
        'test':    ('2018-06-14', '2018-07-15'),
        'test_path': 'data/wc2018.csv',
    },
    2022: {
        'prev_wc': ('2018-06-14', '2018-07-15'),   # WC 2018 tournament
        'qual':    ('2019-01-01', '2022-11-19'),   # strictly before WC 2022 kicks off
        'test':    ('2022-11-20', '2022-12-18'),
        'test_path': 'data/wc2022.csv',
    },
    2026: {
        'prev_wc': ('2022-11-20', '2022-12-18'),   # WC 2022 tournament
        'qual':    ('2023-01-01', '2026-06-10'),   # strictly before WC 2026 kicks off
        'test':    ('2026-06-11', '2026-07-19'),
        'test_path': 'data/wc2026.csv',
    },
}


def _clean(d: pd.DataFrame) -> pd.DataFrame:
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


def fetch_data(year: int = 2022) -> pd.DataFrame:
    """
    Download and filter training data for a given WC year.

    Training window = previous WC tournament + qualifying cycle up to the
    day before the target tournament starts.
    FIFA rankings snapshot used should match this year (set separately in features.py).

    Saves to data/matches.csv and returns the training DataFrame.
    """
    if year not in WC_CONFIG:
        raise ValueError(f"Unsupported year {year}. Supported: {list(WC_CONFIG)}")

    cfg = WC_CONFIG[year]

    print(f"Downloading international results dataset...")
    df = pd.read_csv(SOURCE_URL)

    # ── TRAIN: previous WC tournament ────────────────────────────────────────
    prev_wc = df[
        (df['tournament'] == 'FIFA World Cup') &
        (df['date'] >= cfg['prev_wc'][0]) &
        (df['date'] <= cfg['prev_wc'][1])
    ].copy()

    # ── TRAIN: qualification cycle ────────────────────────────────────────────
    qual = df[
        (df['tournament'].str.contains('FIFA World Cup qualification', case=False, na=False)) &
        (df['date'] >= cfg['qual'][0]) &
        (df['date'] <= cfg['qual'][1])
    ].copy()

    train = _clean(pd.concat([prev_wc, qual], ignore_index=True))
    train.to_csv(TRAIN_PATH, index=False)

    print(f"WC {year} training data → {TRAIN_PATH}")
    print(f"  Previous WC tournament : {len(prev_wc)} matches  ({cfg['prev_wc'][0]} – {cfg['prev_wc'][1]})")
    print(f"  Qualification cycle    : {len(qual)} matches  ({cfg['qual'][0]} – {cfg['qual'][1]})")
    print(f"  Total                  : {len(train)} matches")

    # ── TEST: the WC tournament itself (eval set) ─────────────────────────────
    # Only saved if goals exist in the source (i.e. tournament has been played)
    test_path = cfg.get('test_path')
    if test_path:
        import os
        test_raw = df[
            (df['tournament'] == 'FIFA World Cup') &
            (df['date'] >= cfg['test'][0]) &
            (df['date'] <= cfg['test'][1])
        ].copy()
        test = _clean(test_raw)
        if len(test) > 0:
            test.to_csv(test_path, index=False)
            print(f"  Eval set saved         : {len(test)} matches  → {test_path}")
        else:
            print(f"  Eval set               : 0 matches (tournament not yet played) → {test_path} not written")

    return train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch training data for a World Cup year.')
    parser.add_argument('--year', type=int, default=2022, choices=[2018, 2022, 2026],
                        help='Target World Cup year (default: 2022)')
    args = parser.parse_args()
    fetch_data(year=args.year)
