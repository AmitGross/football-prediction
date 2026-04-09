# Football Prediction — WC 2018, 2022 & 2026

Machine-learning pipeline that predicts international football match scores and simulates full World Cup tournaments. Retrospectively evaluated on WC 2018 and WC 2022; live prediction active for WC 2026 (June 11 – July 19, 2026).

**Model v1.4 · 84 features · Best result: 50% outcome accuracy, 75% knockout accuracy (WC 2022 walk-forward)**

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

## Evaluation results (model v1.4)

### WC 2022 — real Nov 2022 FIFA rankings, 930 training matches

| Mode | Full Tournament | Group Stage | Knockout Rounds | Final |
|------|----------------|-------------|----------------|-------|
| Frozen | 42.2% · RPS 0.235 · RMSE 1.46 | 37.5% | 56.2% | ❌ |
| **Retrain** | **50.0% · RPS 0.216 · RMSE 1.38** | 41.7% | **75.0%** | ✅ |

### WC 2018 — real Jun 7 2018 FIFA rankings, 915 training matches

| Mode | Full Tournament | Group Stage | Knockout Rounds | Final |
|------|----------------|-------------|----------------|-------|
| Frozen | 37.5% · RPS 0.240 · RMSE 1.22 | 35.4% | 43.8% | ✅ |
| Retrain | 40.6% · RPS 0.238 · RMSE 1.22 | 43.8% | 31.2% | ❌ |

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

**84 features per match:**
- Elo ratings + differential
- Kalman filter ratings (attack/defence + uncertainty)
- Form over last 5 and last 2 matches (wins, draws, losses, goals, weighted goals)
- Head-to-head record
- Days rest
- FIFA ranking points (year-correct snapshot)
- PageRank, HITS (hub/authority) graph features with temporal decay
- Neighbourhood aggregation — schedule strength + performance context:
  - `weighted_opp_elo` — outcome-weighted opponent Elo
  - `win_rate_vs_top_teams` — win rate vs opponents in top 30% Elo tier
  - `avg_goal_diff_vs_opp` — average goal difference across all past opponent matches
  - `weighted_goal_diff_by_opp` — goal difference scaled by opponent Elo strength

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

## Current simulation result (April 9, 2026 — model v1.4)

**Predicted champion: Netherlands**  
Path: R32 beat Ivory Coast → R16 beat Germany → QF beat Belgium → SF beat Mexico → **Final vs England** (1–1, Netherlands win)

Top FIFA rankings used (April 1, 2026):  
`France 1877 · Spain 1876 · Argentina 1875 · England 1826 · Portugal 1798 · Brazil 1761`


Machine-learning pipeline that predicts football match scores and simulates full World Cup tournaments.

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

**84 features per match:**
- Elo ratings + differential
- Kalman filter ratings (attack/defence + uncertainty)
- Form over last 5 and last 2 matches (wins, draws, losses, goals, weighted goals)
- Head-to-head record
- Days rest
- FIFA ranking points
- PageRank, HITS (hub/authority) graph features
- Neighbourhood aggregation — schedule strength + performance context:
  - `weighted_opp_elo` — outcome-weighted opponent Elo (+1 win / 0 draw / −1 loss × opp Elo)
  - `win_rate_vs_top_teams` — win rate vs opponents in the top 30% Elo tier
  - `avg_goal_diff_vs_opp` — average goal difference (GF−GA) across all past opponent matches
  - `weighted_goal_diff_by_opp` — goal difference scaled by opponent Elo strength

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

## Current simulation result (April 9, 2026 — model v1.4)

**Predicted champion: Netherlands**  
Path: R32 beat Ivory Coast → R16 beat Germany → QF beat Belgium → SF beat Mexico → **Final vs England** (1–1, Netherlands win)

Top FIFA rankings used (April 1, 2026):  
`France 1877 · Spain 1876 · Argentina 1875 · England 1826 · Portugal 1798 · Brazil 1761`

---

---

## Results

### WC 2022 retrospective evaluation (64 matches, actual scores known)

| Mode | Outcome Accuracy | Mean RPS | RMSE (goals) |
|------|-----------------|----------|---------------|
| **Frozen** | 45.3% | 0.2160 | 1.4031 |
| **Learning (walk-forward)** | **53.1%** | **0.2094** | **1.3607** |

> **RPS benchmark**: < 0.21 = solid · < 0.20 = strong · < 0.195 = excellent  
> **Accuracy benchmark**: > 52% = good for football prediction (high draw rate makes this hard)

#### Knockout stage only — Learning model (16 matches, retrained on all 48 group results)

By the knockout stage the model had been retrained after every group match, fully absorbing 48 real results:

| Metric | Knockout (16 matches) | All 64 matches |
|--------|----------------------|----------------|
| **Outcome accuracy** | **75.0% (12/16)** | 53.1% |
| RMSE (goals) | **1.199** | 1.361 |
| **Mean RPS** | **0.1333** | 0.2094 |

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

### WC 2026 simulation (April 9, 2026 — model v1.4, pre-tournament)

Full tournament simulated from scratch using the frozen model trained on all pre-2026 data + April 1, 2026 official FIFA rankings.

**Group winners (predicted):**

| Group | 1st | 2nd |
|-------|-----|-----|
| A | Mexico | South Korea |
| B | Switzerland | Canada |
| C | Morocco | Brazil |
| D | United States | Turkey |
| E | Germany | Ivory Coast |
| F | Netherlands | Japan |
| G | Belgium | Egypt |
| H | Spain | Uruguay |
| I | France | Norway |
| J | Argentina | Austria |
| K | Portugal | Colombia |
| L | England | Panama |

**Round of 32 (R32) — July 4:**

| Match | Score | Win% | Draw% | Loss% | Advances |
|-------|-------|------|-------|-------|---------|
| Mexico vs Canada | 2-1 | 55.4 | 28.3 | 16.3 | **Mexico** |
| Switzerland vs South Korea | 1-1 | 42.4 | 33.8 | 23.8 | **Switzerland** |
| Morocco vs Turkey | 2-1 | 56.1 | 25.3 | 18.6 | **Morocco** |
| United States vs Brazil | 1-1 | 41.2 | 32.7 | 26.1 | **United States** ⚡ |
| Germany vs Japan | 1-1 | 36.1 | 32.6 | 31.4 | **Germany** |
| Netherlands vs Ivory Coast | 2-1 | 54.9 | 28.6 | 16.5 | **Netherlands** |
| Belgium vs Uruguay | 2-1 | 50.9 | 30.6 | 18.5 | **Belgium** |
| Spain vs Egypt | 2-0 | 59.4 | 27.8 | 12.8 | **Spain** |
| France vs Austria | 2-1 | 56.4 | 26.0 | 17.6 | **France** |
| Argentina vs Norway | 3-1 | 64.5 | 21.7 | 13.7 | **Argentina** |
| Portugal vs Panama | 2-1 | 47.7 | 27.1 | 25.2 | **Portugal** |
| England vs Colombia | 2-1 | 58.8 | 27.3 | 13.9 | **England** |
| Ecuador vs Tunisia | 1-1 | 44.3 | 31.1 | 24.6 | **Ecuador** |
| Croatia vs Bosnia and Herzegovina | 4-0 | 85.6 | 11.6 | 2.7 | **Croatia** |
| Senegal vs Iran | 2-1 | 56.4 | 26.9 | 16.7 | **Senegal** |
| Scotland vs Czech Republic | 2-1 | 52.9 | 28.1 | 19.0 | **Scotland** |

> ⚡ United States eliminates Brazil 1-1 (on pens) at R32 — 41.2% favourites

**Round of 16 (R16) — July 8:**

| Match | Score | Win% | Draw% | Loss% | Advances |
|-------|-------|------|-------|-------|---------|
| Mexico vs Switzerland | 1-1 | 43.7 | 31.3 | 25.0 | **Mexico** |
| Morocco vs United States | 1-1 | 48.1 | 32.1 | 19.8 | **Morocco** |
| Germany vs Netherlands | 1-1 | 32.0 | 34.0 | 34.0 | **Netherlands** ⚡ |
| Belgium vs Spain | 1-1 | 30.8 | 38.4 | 30.8 | **Belgium** ⚡ |
| France vs Argentina | 1-1 | 43.9 | 31.3 | 24.8 | **France** |
| Portugal vs England | 1-1 | 28.2 | 33.9 | 37.9 | **England** ⚡ |
| Ecuador vs Croatia | 1-1 | 35.1 | 37.9 | 27.0 | **Ecuador** ⚡ |
| Senegal vs Scotland | 2-1 | 63.9 | 22.6 | 13.5 | **Senegal** |

> ⚡ Netherlands eliminate Germany (50/50 split) · Belgium knock out Spain (50/50) · England past Portugal (Portugal only 28.2% to win)

**Quarter-Finals (QF) — July 11:**

| Match | Score | Win% | Draw% | Loss% | Winner |
|-------|-------|------|-------|-------|--------|
| Mexico vs Morocco | 1-1 | 32.6 | 35.2 | 32.1 | **🇲🇽 Mexico** ⚡ |
| Netherlands vs Belgium | 1-1 | 42.3 | 34.9 | 22.8 | **🇳🇱 Netherlands** |
| France vs England | 1-1 | 32.2 | 31.7 | 36.0 | **🏴󠁧󠁢󠁥󠁮󠁧󠁿 England** ⚡ |
| Ecuador vs Senegal | 1-1 | 33.7 | 38.3 | 28.0 | **🇪🇨 Ecuador** ⚡ |

> ⚡ England eliminates France (England slight favourites at 36%) · Mexico through another coin-flip QF

**Semi-Finals (SF) — July 14:**

| Match | Score | Win% | Draw% | Loss% | Winner |
|-------|-------|------|-------|-------|--------|
| Mexico vs Netherlands | 1-1 | 30.5 | 34.4 | 35.0 | **🇳🇱 Netherlands** |
| England vs Ecuador | 2-1 | 60.1 | 26.0 | 13.9 | **🏴󠁧󠁢󠁥󠁮󠁧󠁿 England** |

**3rd Place — July 18:** Mexico 2-1 Ecuador

**🏆 Final — July 19, 2026:**

| | Team | | Score | | Team | |
|-|------|---|-------|---|------|---|
| | **Netherlands** | | **1 - 1** | | England | |

> Win probability: **Netherlands 37.9%** · Draw 35.0% · England 27.1%  
> Netherlands edge England in a tight final — the most evenly contested final in recent WC history

**Predicted champion: 🏆 Netherlands**

> This simulation will be updated as real results come in from June 11, 2026 onward.  
> Full bracket file: [`predictions_wc2026_full_v1.4.xlsx`](predictions_wc2026_full_v1.4.xlsx)

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