"""
update_rankings.py
------------------
Applies the April 1, 2026 official FIFA rankings to data/fifa_rankings.csv.
Run this script whenever new rankings are published and you want to refresh
the model features.

Source: FIFA/Coca-Cola Men's World Ranking (official update: 01 April 2026)
        https://inside.fifa.com/fifa-world-ranking/men
Next official update: 10 June 2026

Usage:
    python update_rankings.py
"""

import pandas as pd
import os

# -------------------------------------------------------------------------
# April 1, 2026 – FIFA Men's World Ranking (official update)
# Exact values for top 20 from FIFA/Wikipedia; integer values for ranks 21+
# from Transfermarkt (which rounds to nearest integer).
# -------------------------------------------------------------------------
APRIL_2026_RANKINGS = {
    # Top 20 (exact, from official FIFA source)
    "France":                1877.32,
    "Spain":                 1876.40,
    "Argentina":             1874.81,
    "England":               1825.97,
    "Portugal":              1763.83,
    "Brazil":                1761.16,
    "Netherlands":           1757.87,
    "Morocco":               1755.87,
    "Belgium":               1734.71,
    "Germany":               1730.37,
    "Croatia":               1717.07,
    "Italy":                 1700.37,
    "Colombia":              1693.09,
    "Senegal":               1688.99,
    "Mexico":                1681.03,
    "United States":         1673.13,
    "Uruguay":               1673.07,
    "Japan":                 1660.43,
    "Switzerland":           1649.40,
    "Denmark":               1620.81,
    # Ranks 21-100 (from Transfermarkt, rounded integer)
    "Iran":                  1615.0,
    "Turkey":                1599.0,
    "Ecuador":               1595.0,
    "Austria":               1593.0,
    "South Korea":           1589.0,
    "Nigeria":               1585.0,
    "Australia":             1581.0,
    "Algeria":               1564.0,
    "Egypt":                 1563.0,
    "Canada":                1556.0,
    "Norway":                1551.0,
    "Ukraine":               1547.0,
    "Panama":                1541.0,
    "Ivory Coast":           1533.0,
    "Poland":                1528.0,
    "Russia":                1526.0,
    "Wales":                 1524.0,
    "Sweden":                1515.0,
    "Serbia":                1509.0,
    "Paraguay":              1504.0,
    "Czech Republic":        1501.0,
    "Hungary":               1501.0,
    "Scotland":              1498.0,
    "Tunisia":               1483.0,
    "Cameroon":              1481.0,
    "DR Congo":              1478.0,
    "Greece":                1476.0,
    "Slovakia":              1474.0,
    "Venezuela":             1468.0,
    "Uzbekistan":            1465.0,
    "Costa Rica":            1460.0,
    "Mali":                  1459.0,
    "Peru":                  1456.0,
    "Chile":                 1455.0,
    "Qatar":                 1455.0,
    "Romania":               1451.0,
    "Iraq":                  1447.0,
    "Slovenia":              1446.0,
    "Republic of Ireland":   1437.0,
    "South Africa":          1430.0,
    "Saudi Arabia":          1421.0,
    "Burkina Faso":          1412.0,
    "Jordan":                1391.0,
    "Albania":               1388.0,
    "Bosnia and Herzegovina":1386.0,
    "Honduras":              1380.0,
    "North Macedonia":       1372.0,
    "United Arab Emirates":  1370.0,
    "Cape Verde":            1366.0,
    "Northern Ireland":      1362.0,
    "Jamaica":               1358.0,
    "Georgia":               1350.0,
    "Finland":               1346.0,
    "Ghana":                 1346.0,
    "Iceland":               1345.0,
    "Bolivia":               1329.0,
    "Israel":                1328.0,
    "Kosovo":                1319.0,
    "Oman":                  1313.0,
    "Guinea":                1300.0,
    "Montenegro":            1296.0,
    "Curacao":               1295.0,
    "Haiti":                 1292.0,
    "Syria":                 1289.0,
    "New Zealand":           1282.0,
    "Bulgaria":              1279.0,
    "Gabon":                 1273.0,
    "Uganda":                1264.0,
    "Angola":                1263.0,
    "Benin":                 1259.0,
    "Bahrain":               1259.0,
    "Zambia":                1256.0,
    "Thailand":              1252.0,
    "China PR":              1252.0,
    "Palestine":             1245.0,
    "Guatemala":             1243.0,
    "Belarus":               1236.0,
    "Luxembourg":            1228.0,
    "Vietnam":               1226.0,
    "El Salvador":           1225.0,
}

RANKING_DATE = "2026-04-01"

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "fifa_rankings.csv")

    # Read existing rankings
    df = pd.read_csv(csv_path)
    original_count = len(df)

    updated = 0
    added = 0

    # Apply updates
    for team, points in APRIL_2026_RANKINGS.items():
        mask = df["team"] == team
        if mask.any():
            old_val = df.loc[mask, "fifa_points"].values[0]
            df.loc[mask, "fifa_points"] = points
            if abs(old_val - points) > 0.01:
                print(f"  Updated: {team:35s} {old_val:.1f} -> {points:.2f}")
                updated += 1
        else:
            new_row = pd.DataFrame({"team": [team], "fifa_points": [points]})
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"  Added:   {team:35s} {points:.2f}")
            added += 1

    # Write back with metadata comment in a sidecar file
    df.to_csv(csv_path, index=False)

    # Write metadata
    meta_path = os.path.join(os.path.dirname(__file__), "data", "fifa_rankings_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"last_updated={RANKING_DATE}\n")
        f.write(f"source=FIFA/Coca-Cola Men's World Ranking (official)\n")
        f.write(f"url=https://inside.fifa.com/fifa-world-ranking/men\n")
        f.write(f"next_update=2026-06-10\n")

    print(f"\nDone. {updated} teams updated, {added} teams added.")
    print(f"Total teams in file: {len(df)} (was {original_count})")
    print(f"Ranking date: {RANKING_DATE}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    main()
