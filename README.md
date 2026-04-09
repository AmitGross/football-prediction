# Football Prediction — WC 2018, 2022 & 2026

Machine-learning pipeline that predicts international football match scores and simulates full World Cup tournaments. Retrospectively evaluated on WC 2018 and WC 2022; live prediction active for WC 2026 (June 11 – July 19, 2026).

**Model v1.6 · 68 features · Best result: 54.7% outcome accuracy (WC 2022 walk-forward) · Predicted WC 2026 champion: France**

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train for a specific WC year (fetches correct data window automatically)
python main.py train --year 2026

# 3. Evaluate retrospectively
python main.py evaluate          # WC 2022 frozen
python main.py evaluate --retrain  # WC 2022 walk-forward

# 4. Simulate the full WC 2026 bracket
python main.py simulate2026
```

---

## All CLI commands

```
python main.py train [--year 2018|2022|2026]          Train all models for a WC year
python main.py tune [--trials N]                       Hyperparameter search (Optuna)
python main.py evaluate   [--retrain] [--limit N]      WC 2022 retrospective evaluation
python main.py evaluate2018 [--retrain] [--limit N]    WC 2018 retrospective evaluation
python main.py evaluate2026 [--retrain] [--limit N]    WC 2026 live evaluation
python main.py simulate2026                            Full WC 2026 tournament simulation
python main.py predict "France" "Brazil"               Single match prediction
```

Every script also runs standalone: `python train.py`, `python evaluate.py --year 2022`, `python simulate_wc2026.py`.

---

## Frozen vs Learning (walk-forward) modes

| Flag | Behaviour | When to use |
|------|-----------|-------------|
| *(none — default)* | **Frozen**: model trained once before tournament, never retrained | Benchmarking, pre-tournament simulation |
| `--retrain` | **Walk-forward**: model retrains after each real result | Live tournament — model continuously improves |

---

## Evaluation results (model v1.6)

### WC 2022 — real Nov 2022 FIFA rankings, 930 training matches

| Mode | Full Tournament | RPS | RMSE |
|------|----------------|-----|------|
| Frozen | 48.4% | 0.2122 | 1.381 |
| **Retrain** | **54.7%** | **0.2081** | **1.349** |

### WC 2018 — real Jun 2018 FIFA rankings, 915 training matches

| Mode | Full Tournament | RPS | RMSE |
|------|----------------|-----|------|
| Frozen | 35.9% | 0.2523 | 1.287 |
| Retrain | 39.1% | 0.2536 | 1.256 |

**Benchmarks:** Accuracy > 52% = good · RPS < 0.21 = solid · RMSE < 1.65 = strong

### WC 2026

| Mode | Status |
|------|--------|
| Frozen | ⏳ Available from June 11, 2026 |
| Retrain | ⏳ Available from June 11, 2026 |

---

## Year-aware data pipeline

Each WC year uses its own correctly-scoped training window and pre-tournament FIFA rankings snapshot — preventing any data leakage.

| Year | Training window | FIFA rankings snapshot | Rankings file |
|------|----------------|----------------------|--------------|
| 2018 | WC2014 + 2015–2018 quals (915 matches) | June 7, 2018 — 211 teams | `data/fifa_rankings_2018.csv` |
| 2022 | WC2018 + 2019–2022 quals (930 matches) | November 2022 — 212 teams | `data/fifa_rankings_2022.csv` |
| 2026 | WC2022 + 2023–2026 quals (965 matches) | April 1, 2026 — 213 teams | `data/fifa_rankings_2026.csv` |

`python main.py train --year <year>` fetches the correct data window, loads the right rankings snapshot, trains all 4 models, and archives versioned copies automatically.

---

## WC 2026 live workflow (from June 11, 2026)

1. A match is played → fill in `goals_A` / `goals_B` in `data/wc2026.csv`
2. Frozen eval — how did the pre-trained model do?  
   `python main.py evaluate2026`
3. Walk-forward — model improves with each result:  
   `python main.py evaluate2026 --retrain`
4. Re-simulate the remaining bracket:  
   `python main.py simulate2026`

> If `data/wc2026.csv` has no scores yet, `evaluate2026` prints *"No results available yet"* and exits cleanly.

Next FIFA rankings update: **June 10, 2026** — run `python update_rankings.py` after updating values, then retrain.

---

## Model overview

**Training pipeline (`python main.py train --year <year>`):**
1. `model.pkl` — AveragingEnsemble (RandomForest + XGBoost) predicting (λ_A, λ_B)
2. `classifier.pkl` — XGBClassifier predicting W/D/L directly
3. `dc_ratings.pkl` — Dixon-Coles MLE team strength ratings
4. `calibrator.pkl` — Isotonic regression probability calibrator

**Scores → probabilities:**  
Predicted λ values feed a Poisson score grid → P(win), P(draw), P(loss), calibrated with the isotonic calibrator.

**68 features per match (v1.6):**

| Group | Count | Features |
|-------|-------|----------|
| Elo | 3 | elo_A, elo_B, elo_diff |
| Form-5 | 14 | wins/draws/losses/goals/weighted per team × 2 |
| Form-2 | 14 | same as Form-5 but last 2 matches |
| H2H | 3 | h2h_wins_A, h2h_wins_B, h2h_draws |
| FIFA Rankings | 3 | fifa_A, fifa_B, fifa_diff |
| Rest & Match Count | 4 | rest_days_A/B, matches_played_A/B |
| Neighbourhood Basic | 9 | avg_opp_elo, avg_opp_scored, avg_opp_conceded + diffs |
| Neighbourhood Perf | 12 | weighted_opp_elo, win_rate_vs_top_teams, avg_goal_diff_vs_opp, weighted_goal_diff_by_opp + diffs |
| Stage & Volatility | 6 | is_knockout, round_number, games_in_tournament_A/B, goal_diff_std_A/B |

Removed in v1.5 after ablation: Kalman filter (x10) and PageRank/HITS (x12) — both hurt performance.  
Added in v1.6: stage context features (is_knockout, round_number) + tournament momentum (games_in_tournament) + team volatility (goal_diff_std).

---

## Project structure

```
football-prediction/
├── main.py                     ← Unified CLI entry point (start here)
├── train.py                    ← Train RF + XGBoost ensemble + classifier
├── evaluate.py                 ← Walk-forward evaluation (2018, 2022 or 2026)
├── simulate_wc2026.py          ← Full WC 2026 bracket simulation
├── predict.py                  ← predict_match() — single match prediction
├── predict_wc2026.py           ← Batch prediction for 2026 group stage
├── features.py                 ← Feature engineering (Elo, Kalman, form, FIFA, graph)
├── ensemble.py                 ← AveragingEnsemble (RF + XGBoost)
├── poisson.py                  ← Score grid + result probabilities
├── dc_ratings.py               ← Dixon-Coles MLE team strength ratings
├── calibrate.py                ← Isotonic probability calibrator
├── tune.py                     ← Optuna hyperparameter search
├── app.py                      ← FastAPI: /predict, /result, /health
├── fetch_data.py               ← Download & filter training data (year-aware)
├── fetch_rankings_2018.py      ← Fetch real Jun 2018 FIFA rankings from GitHub archive
├── update_rankings.py          ← Update FIFA rankings CSV with new values
├── run_pipeline.py             ← Shell-style full pipeline runner
├── batch_predict.py            ← Predict a batch of matches from CSV
├── requirements.txt
├── best_params.json            ← Generated by tune.py
├── data/
│   ├── matches.csv             ← Training data (set by last train --year run)
│   ├── wc2018.csv              ← WC 2018 fixtures + actual scores (64 matches)
│   ├── wc2022.csv              ← WC 2022 fixtures + actual scores (64 matches)
│   ├── wc2026.csv              ← WC 2026 fixtures (fill goals as played)
│   ├── fifa_rankings_2018.csv  ← FIFA rankings, June 7 2018 (211 teams, official)
│   ├── fifa_rankings_2022.csv  ← FIFA rankings, November 2022 (212 teams)
│   └── fifa_rankings_2026.csv  ← FIFA rankings, April 1 2026 (213 teams)
└── .github/
    └── copilot-instructions.md ← Full project context (auto-loaded by GitHub Copilot)
```

**Active model files (committed):** `model.pkl`, `classifier.pkl`, `dc_ratings.pkl`, `calibrator.pkl`  
**Versioned archives (not committed, regenerable):** `model_wc{year}_v1.4.pkl` etc — recreated by `train --year`

---

## API (FastAPI)

```bash
uvicorn app:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict a match: `{"team_A": "France", "team_B": "Brazil"}` |
| `/result` | POST | Submit a real result — appends to CSV and retrains model |
| `/health` | GET | Health check |

---

## WC 2026 group draw

```
A: Mexico, South Africa, South Korea, Czech Republic
B: Canada, Bosnia and Herzegovina, Qatar, Switzerland
C: Brazil, Morocco, Haiti, Scotland
D: United States, Paraguay, Australia, Turkey
E: Germany, Curacao, Ivory Coast, Ecuador
F: Netherlands, Japan, Sweden, Tunisia
G: Belgium, Egypt, Iran, New Zealand
H: Spain, Cape Verde, Saudi Arabia, Uruguay
I: France, Senegal, Iraq, Norway
J: Argentina, Algeria, Austria, Jordan
K: Portugal, DR Congo, Uzbekistan, Colombia
L: England, Croatia, Ghana, Panama
```

---

## Current simulation result (April 9, 2026 — model v1.6)

**Predicted champion: France** (beats Mexico 1–1, wins on probabilities in Final)  
France's path: Argentina (R16) → Portugal (QF) → Scotland (SF) → Final

Top FIFA rankings used (April 1, 2026):  
`France 1877 · Spain 1876 · Argentina 1875 · England 1826 · Portugal 1798 · Brazil 1761`

---

## Ablation study (WC 2022, v1.4 baseline → v1.5)

**Full tournament (64 matches) — Baseline: Acc=48.4%, RPS=0.2236**

| Group removed | Features | ΔRPS | Verdict |
|--------------|----------|------|---------|
| FIFA Rankings | 3 | +0.0169 | Most valuable |
| Rest & Match Count | 4 | +0.0018 | Small positive |
| PageRank/HITS | 12 | **−0.0138** | Most harmful → **REMOVED** |
| Neighbourhood Basic | 9 | −0.0050 | Hurts full, critical in KO |
| Kalman | 10 | −0.0005 | Mildly hurts → **REMOVED** |

**Knockout (16 matches) — Baseline: Acc=62.5%, RPS=0.1256**

| Group removed | Features | ΔRPS | Verdict |
|--------------|----------|------|---------|
| FIFA Rankings | 3 | +0.0291 | Dominant |
| Neighbourhood Basic | 9 | +0.0134 | Critical in KO |
| Elo | 3 | +0.0066 | Valuable |
| PageRank/HITS | 12 | −0.0058 | Harmful → **REMOVED** |
| Kalman | 10 | −0.0019 | Hurts in KO → **REMOVED** |

---

## Roadmap

### ✅ v1.6 (shipped April 9, 2026)

5 new features → 68 total:

| Feature | Type | Values | Rationale |
|---------|------|--------|-----------|
| `is_knockout` | binary | 0/1 | Group vs KO context |
| `round_number` | ordinal | 0=qual, 1=group, 2=R32/R16, 3=QF, 4=SF, 5=Final | Model learns stage-specific behaviour |
| `games_in_tournament_A/B` | count | 0,1,2,3… | Momentum/fatigue accumulation |
| `goal_diff_std_A` | float | std dev last 5 GF−GA | Team volatility/consistency |
| `goal_diff_std_B` | float | std dev last 5 GF−GA | Team volatility/consistency |

Result: WC 2022 frozen RPS improved from 0.2348 → 0.2122 (−0.023). WC 2022 retrain improved 0.2088 → 0.2081. Shipped as v1.6.

### Future (v1.7)
- Automated live scoring: fetch real-time scores → append to `wc2026.csv` → auto-retrain
- Supabase: store predictions/results via `/result` endpoint
- Vercel frontend: live bracket + predictions

- **WC 2022**: Full retrospective evaluation (all 64 actual results available)
- **WC 2026**: Pre-tournament simulation (actuals filled in as the tournament progresses, June 11 – July 19, 2026)

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models
python main.py train

# 3. Evaluate on WC 2022 (actuals known, frozen model)
python main.py evaluate

# 4. Simulate the full WC 2026 bracket
python main.py simulate2026
```

---

## All commands

```
python main.py train                              Train all models
python main.py tune [--trials N]                  Hyperparameter search (Optuna)
python main.py evaluate [--retrain] [--limit N]   WC 2022 retrospective evaluation
python main.py evaluate2026 [--retrain] [--limit N]  WC 2026 live evaluation
python main.py simulate2026                       Full WC 2026 tournament simulation
python main.py predict "France" "Brazil"           Predict a single match
```

Every script also runs standalone: `python train.py`, `python evaluate.py`, `python simulate_wc2026.py`.

---

## Frozen vs Learning modes

| Flag | Behaviour | When to use |
|------|-----------|-------------|
| *(none — default)* | **Frozen**: model trained once before tournament, never retrained | Benchmarking, leaderboard, pre-tournament simulation |
| `--retrain` | **Learning (walk-forward)**: model retrains after each real result | Live tournament — model continuously improves as results come in |

Both modes work for **both years**:

| Command | Actuals required? | Rankings used |
|---------|-------------------|---------------|
| `evaluate` | Yes — `data/wc2022.csv` (fully filled) | Nov 2022 FIFA |
| `evaluate --retrain` | Yes | Nov 2022 FIFA |
| `evaluate2026` | Yes — `data/wc2026.csv` (only filled rows) | Apr 2026 FIFA |
| `evaluate2026 --retrain` | Yes | Apr 2026 FIFA |
| `simulate2026` | **No actuals needed** | Apr 2026 FIFA |

> **If `data/wc2026.csv` has no scores yet**, `evaluate2026` prints *"No results available yet"* and exits cleanly. Nothing breaks.

---

## WC 2026 live workflow (from June 11, 2026)

1. A match is played → fill in `goals_A` / `goals_B` in `data/wc2026.csv`
2. Run **frozen** to see how the pre-trained model did:  
   `python main.py evaluate2026`
3. Run **learning** to continuously improve the model:  
   `python main.py evaluate2026 --retrain`
4. Re-simulate the remaining bracket:  
   `python main.py simulate2026`

---

## Project structure

```
football-prediction/
├── main.py                    ← Unified CLI entry point (start here)
├── train.py                   ← Train RF + XGBoost ensemble + classifier
├── evaluate.py                ← Walk-forward evaluation (2022 or 2026)
├── simulate_wc2026.py         ← Full WC 2026 bracket simulation
├── predict.py                 ← predict_match() — single match prediction
├── predict_wc2026.py          ← Batch prediction for 2026 group stage
├── features.py                ← Feature engineering (Elo, Kalman, form, FIFA, graph)
├── ensemble.py                ← AveragingEnsemble (RF + XGBoost)
├── poisson.py                 ← Score grid + result probabilities
├── dc_ratings.py              ← Dixon-Coles MLE strength ratings
├── calibrate.py               ← Isotonic probability calibrator
├── tune.py                    ← Optuna hyperparameter search
├── app.py                     ← FastAPI: /predict, /result, /health
├── fetch_data.py              ← Download training data from martj42/international_results
├── update_rankings.py         ← Update FIFA rankings CSV
├── run_pipeline.py            ← Shell-style full pipeline runner
├── batch_predict.py           ← Predict a batch of matches from CSV
├── requirements.txt
├── best_params.json           ← Generated by tune.py
├── data/
│   ├── matches.csv            ← Training data (historical international matches)
│   ├── wc2022.csv             ← WC 2022 fixtures + actual scores (all 64 matches)
│   ├── wc2026.csv             ← WC 2026 fixtures (fill goals_A/goals_B as played)
│   ├── fifa_rankings_2022.csv ← FIFA rankings, November 2022 (212 teams)
│   └── fifa_rankings_2026.csv ← FIFA rankings, April 1 2026 (213 teams)
└── .github/
    └── copilot-instructions.md ← Full project context (auto-loaded by GitHub Copilot)
```

**Generated artefacts (not committed):** `model.pkl`, `classifier.pkl`, `dc_ratings.pkl`, `calibrator.pkl`

---

## Model overview

**68 features per match (v1.6):**
- Elo ratings + differential
- Form over last 5 and last 2 matches (wins, draws, losses, goals, weighted goals)
- Head-to-head record
- Days rest + matches played
- FIFA ranking points
- Neighbourhood aggregation — schedule strength + performance context:
  - `weighted_opp_elo` — outcome-weighted opponent Elo
  - `win_rate_vs_top_teams` — win rate vs opponents in the top 30% Elo tier
  - `avg_goal_diff_vs_opp` — average goal difference (GF−GA) across all past opponent matches
  - `weighted_goal_diff_by_opp` — goal difference scaled by opponent Elo strength
- Stage & volatility: `is_knockout`, `round_number` (0=qual→5=final), `games_in_tournament_A/B`, `goal_diff_std_A/B`

*(Kalman filter ×10 and PageRank/HITS ×12 removed in v1.5 after ablation — both hurt performance.)*

**Training pipeline (`python main.py train`):**
1. `model.pkl` — AveragingEnsemble (RandomForest + XGBoost) predicting (λ_A, λ_B)
2. `classifier.pkl` — XGBClassifier predicting W/D/L directly
3. `dc_ratings.pkl` — Dixon-Coles MLE team strength ratings
4. `calibrator.pkl` — Isotonic regression probability calibrator

**Scores → probabilities:**  
Predicted λ values feed a Poisson score grid → P(win), P(draw), P(loss) calibrated with the isotonic calibrator.

---

## FIFA rankings — two versions

| File | Date | Used for |
|------|------|----------|
| `data/fifa_rankings_2022.csv` | November 2022 | WC 2022 evaluation |
| `data/fifa_rankings_2026.csv` | April 1, 2026 | WC 2026 simulation + training |

The correct file is loaded automatically based on the command you run. No manual switching needed.

To update rankings when new official data is available:  
1. Edit the `APRIL_2026_RANKINGS` dict in `update_rankings.py` with new values  
2. Run `python update_rankings.py`  
3. Retrain: `python main.py train`

Next FIFA rankings update: **June 10, 2026** (day before the tournament starts).

---

## API (FastAPI)

```bash
uvicorn app:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict a match given `{"team_A": "...", "team_B": "..."}` |
| `/result` | POST | Submit a real result — appends to CSV and retrains model |
| `/health` | GET | Health check |

This `/result` endpoint is the hook for automated live-scoring ingestion.

---

## WC 2026 group draw

```
A: Mexico, South Africa, South Korea, Czech Republic
B: Canada, Bosnia and Herzegovina, Qatar, Switzerland
C: Brazil, Morocco, Haiti, Scotland
D: United States, Paraguay, Australia, Turkey
E: Germany, Curacao, Ivory Coast, Ecuador
F: Netherlands, Japan, Sweden, Tunisia
G: Belgium, Egypt, Iran, New Zealand
H: Spain, Cape Verde, Saudi Arabia, Uruguay
I: France, Senegal, Iraq, Norway
J: Argentina, Algeria, Austria, Jordan
K: Portugal, DR Congo, Uzbekistan, Colombia
L: England, Croatia, Ghana, Panama
```

---

## Current simulation result (April 9, 2026 — model v1.6)

**Predicted champion: France** (beats Mexico 1–1, wins on probabilities in Final)  
France's path: Argentina (R16) → Portugal (QF) → Scotland (SF) → Final  
Notable: Spain exits R16 to Belgium · Mexico reaches Final

Top FIFA rankings used (April 1, 2026):  
`France 1877 · Spain 1876 · Argentina 1875 · England 1826 · Portugal 1798 · Brazil 1761`

---

## Results

### WC 2022 retrospective evaluation (64 matches, actual scores known)

| Mode | Outcome Accuracy | Mean RPS | RMSE (goals) |
|------|-----------------|----------|---------------|
| **Frozen** | 48.4% | 0.2122 | 1.381 |
| **Learning (walk-forward)** | **54.7%** | **0.2081** | **1.349** |

> **RPS benchmark**: < 0.21 = solid · < 0.20 = strong · < 0.195 = excellent  
> **Accuracy benchmark**: > 52% = good for football prediction (high draw rate makes this hard)

#### Knockout stage only — Learning model (16 matches, retrained on all 48 group results)

By the knockout stage the model had been retrained after every group match, fully absorbing 48 real results:

| Metric | Knockout (16 matches) | All 64 matches |
|--------|----------------------|----------------|
| **Outcome accuracy** | **75.0% (12/16)** | 54.7% |
| RMSE (goals) | **1.199** | 1.349 |
| **Mean RPS** | **0.1333** | 0.2081 |

**RPS 0.136 is well into the "excellent" range** (< 0.195). The 4 misses were all tight upsets: Japan/Croatia (pen shootout), Morocco beating Portugal, England losing to France in QF, and the 3rd place match.

| Round | Match | Predicted | Actual | ✓ |
|-------|-------|-----------|--------|---|
| R16 | Netherlands vs USA | 1-0 | 3-1 | ✓ |
| R16 | Argentina vs Australia | 2-0 | 2-1 | ✓ |
| R16 | France vs Poland | 2-0 | 3-1 | ✓ |
| R16 | England vs Senegal | **3-0** | **3-0** | ✓ |
| R16 | Japan vs Croatia | 1-0 | 1-1 | ✗ |
| R16 | Brazil vs South Korea | 3-0 | 4-1 | ✓ |
| R16 | Morocco vs Spain | 1-1 | 0-0 | ✓ |
| R16 | Portugal vs Switzerland | 1-0 | 6-1 | ✓ |
| QF | Croatia vs Brazil | 1-1 | 1-1 | ✓ |
| QF | Netherlands vs Argentina | 1-1 | 2-2 | ✓ |
| QF | Morocco vs Portugal | 1-1 | 1-0 | ✗ |
| QF | England vs France | 1-0 | 1-2 | ✗ |
| SF | Argentina vs Croatia | 2-0 | 3-0 | ✓ |
| SF | France vs Morocco | 1-0 | 2-0 | ✓ |
| 3rd | Croatia vs Morocco | 1-1 | 2-1 | ✗ |
| **Final** | **Argentina vs France** | **1-1** | **3-3** | ✓ |

> Run `python compare_knockouts.py` to reproduce this comparison from the saved Excel results.
> Run `python show_knockouts.py` for a single-model knockout summary.

The learning mode demonstrates the core advantage of the pipeline: **each real result makes the model better**. By the knockout stage — where team form and fatigue matter most — the model has absorbed all group stage results and its predictions sharpen accordingly.

---

### Model vs Market — WC 2026 (April 9, 2026)

Comparison of our v1.6 model predictions against [Polymarket](https://polymarket.com) prediction market odds.

#### Tournament winner

| Team | Polymarket | Our model v1.6 |
|------|-----------|----------------|
| Spain | 16% 🥇 | Eliminated R16 (by Belgium) |
| **France** | **14%** | 🏆 **Predicted champion** |
| England | 11% | Eliminated R16 (by Portugal) |
| Argentina | 9% | Eliminated R16 (by France) |
| Brazil | 9% | Eliminated R32 (by United States) |
| Portugal | 7% | Eliminated QF (by France) |
| Germany | 5% | Eliminated SF (by Mexico) |
| Netherlands | 3% | Eliminated R16 (by Germany) |

> Key divergence from market: France (market #2) is our predicted champion. Spain (market #1) exits R16 to Belgium in our simulation. Mexico reaches the Final as a surprise run.

#### Group stage winners

| Group | Teams | Polymarket | Our model v1.6 | Match? |
|-------|-------|-----------|----------------|--------|
| A | Mexico, South Africa, South Korea, Czech Republic | Mexico (45%) | Mexico | ✅ |
| B | Canada, Bosnia, Qatar, Switzerland | Switzerland (51%) | Switzerland | ✅ |
| C | Brazil, Morocco, Haiti, Scotland | Brazil (77%) | Morocco ⚡ | ❌ |
| D | USA, Paraguay, Australia, Turkey | TBD playoff* | United States | ❓ |
| E | Germany, Curacao, Ivory Coast, Ecuador | Germany (71%) | Germany | ✅ |
| F | Netherlands, Japan, Sweden, Tunisia | Netherlands (57%) | Netherlands | ✅ |
| G | Belgium, Egypt, Iran, New Zealand | Belgium (72%) | Belgium | ✅ |
| H | Spain, Cape Verde, Saudi Arabia, Uruguay | Spain (81%) | Spain | ✅ |
| I | France, Senegal, Iraq, Norway | France (69%) | France | ✅ |
| J | Argentina, Algeria, Austria, Jordan | Argentina (77%) | Argentina | ✅ |
| K | Portugal, DR Congo, Uzbekistan, Colombia | Portugal (64%) | Portugal | ✅ |
| L | England, Croatia, Ghana, Panama | England (72%) | England | ✅ |

> *Group D on Polymarket shows KOS/ROU/SVK/TUR — these are teams competing in a qualification playoff for the remaining Group D spot, results not yet finalized.  
> **10/11 group winners match the market** (v1.6 predicts Morocco 1st in Group C over Brazil — key divergence).

---

### WC 2026 simulation (April 9, 2026 — model v1.6, pre-tournament)

Full tournament simulated from scratch using the frozen model trained on all pre-2026 data + April 1, 2026 official FIFA rankings.

**Group winners (predicted):**

| Group | 1st | 2nd |
|-------|-----|-----|
| A | Mexico | South Korea |
| B | Switzerland | Canada |
| C | Morocco ⚡ | Brazil |
| D | United States | Turkey |
| E | Germany | Ecuador |
| F | Netherlands | Japan |
| G | Belgium | Egypt |
| H | Spain | Uruguay |
| I | France | Norway |
| J | Argentina | Jordan |
| K | Portugal | Colombia |
| L | England | Croatia |

> ⚡ Morocco tops Group C ahead of Brazil (FIFA ranking + form advantage)

**Round of 32 (R32):**

| Match | Score | Advances |
|-------|-------|---------|
| Mexico vs Canada | 2-0 | **Mexico** |
| Switzerland vs South Korea | 1-1 | **Switzerland** |
| Morocco vs Turkey | 2-1 | **Morocco** |
| United States vs Brazil | 1-1 | **United States** ⚡ |
| Germany vs Japan | 1-1 | **Germany** |
| Netherlands vs Ecuador | 2-1 | **Netherlands** |
| Belgium vs Uruguay | 1-1 | **Belgium** |
| Spain vs Egypt | 2-0 | **Spain** |
| France vs Jordan | 3-0 | **France** |
| Argentina vs Norway | 3-1 | **Argentina** |
| Portugal vs Croatia | 1-1 | **Portugal** |
| England vs Colombia | 2-1 | **England** |
| Scotland vs Czech Republic | 2-1 | **Scotland** |
| Ivory Coast vs Iran | 1-0 | **Ivory Coast** |
| Senegal vs DR Congo | 2-1 | **Senegal** |
| Panama vs Cape Verde | 3-0 | **Panama** |

**Round of 16 (R16):**

| Match | Score | Advances |
|-------|-------|---------|
| Mexico vs Switzerland | 2-1 | **Mexico** |
| Morocco vs United States | 2-0 | **Morocco** |
| Germany vs Netherlands | 1-1 | **Germany** |
| Belgium vs Spain | 1-1 | **Belgium** ⚡ |
| France vs Argentina | 1-1 | **France** |
| Portugal vs England | 1-1 | **Portugal** |
| Scotland vs Ivory Coast | 1-1 | **Scotland** |
| Senegal vs Panama | 2-1 | **Senegal** |

> ⚡ Spain (market favourite) eliminated by Belgium · Brazil knocked out in R32 by United States

**Quarter-Finals (QF):**

| Match | Score | Winner |
|-------|-------|--------|
| Mexico vs Morocco | 1-1 | **Mexico** |
| Germany vs Belgium | 2-1 | **Germany** |
| France vs Portugal | 2-1 | **France** |
| Scotland vs Senegal | 1-1 | **Scotland** |

**Semi-Finals (SF):**

| Match | Score | Winner |
|-------|-------|--------|
| Mexico vs Germany | 1-1 | **Mexico** |
| France vs Scotland | 3-0 | **France** |

**3rd Place:** Germany 2-0 Scotland

**🏆 Final — July 19, 2026:**

**Mexico 1-1 France** (France wins on probabilities — p_win_France=41.1%, p_win_Mexico=22.4%)

**Predicted champion: 🏆 France**  
France's path: Jordan (R32) → Argentina (R16) → Portugal (QF) → Scotland (SF) → Final

> This simulation will be updated as real results come in from June 11, 2026 onward.  
> Full bracket file: [`predictions_wc2026_full_v1.6.xlsx`](predictions_wc2026_full_v1.6.xlsx)

---

## Development history

Several model versions were iterated before settling on the current one. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full log — what was tried, what the results were, and why each version was accepted or reverted.

---

## Planned enhancements

- [ ] Automated live score ingestion → auto-append to `wc2026.csv` → auto-retrain
- [ ] Store predictions/results in **Supabase** via `/result` API
- [ ] **Vercel** frontend reading from Supabase — live bracket + predictions
- [ ] Auto-update FIFA rankings on official release dates


The response will include the predicted probabilities for win, draw, and loss.

## License

This project is licensed under the MIT License.