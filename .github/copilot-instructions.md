# Football Prediction — GitHub Copilot Project Context

This file is auto-loaded by GitHub Copilot in every conversation. Read this fully before helping.

---

## What this project does

A **machine-learning pipeline** that predicts football (soccer) match scores and simulates full World Cup tournaments. Currently configured for WC 2022 retrospective evaluation and WC 2026 live prediction.

---

## Tech stack

- **Language**: Python 3.10+, conda env `soccer_game_ml_`
- **ML**: RandomForestRegressor + XGBRegressor (AveragingEnsemble for λ prediction), XGBClassifier (W/D/L)
- **Other models**: Dixon-Coles MLE strength ratings, Poisson score-grid probabilities, isotonic calibrator
- **Features (72 total)**: Elo ratings, Kalman filter ratings, form (last 5 + last 2), H2H, days rest, FIFA rankings, PageRank/HITS graph features, neighbourhood aggregation
- **API**: FastAPI (`app.py`) with `/predict` and `/result` endpoints
- **Deterministic**: `random_state=42` everywhere

## Run commands (always use this form)

```powershell
conda run -n soccer_game_ml_ --cwd "C:\Users\user\anaconda3\envs\soccer_game_ml_code\football-prediction" --no-capture-output python <script.py> [args]
```

---

## File map

| File | Purpose |
|------|---------|
| `main.py` | **Unified CLI** — all commands go through here |
| `train.py` | Train RF + XGBoost ensemble → `model.pkl`, trains classifier → `classifier.pkl` |
| `evaluate.py` | Walk-forward retrospective eval (WC 2022 or WC 2026 live) |
| `features.py` | Feature engineering — Elo, Kalman, form, H2H, FIFA rankings, graph features |
| `predict.py` | `predict_match(team_A, team_B, history)` → goals + probabilities |
| `simulate_wc2026.py` | Full WC 2026 bracket simulation (group stage → final) |
| `predict_wc2026.py` | Batch prediction for all WC 2026 group-stage matches |
| `tune.py` | Optuna hyperparameter search → `best_params.json` |
| `poisson.py` | Score grid + result probabilities from predicted λ values |
| `ensemble.py` | `AveragingEnsemble` wrapping RF + XGBoost |
| `dc_ratings.py` | Dixon-Coles MLE team strength ratings |
| `calibrate.py` | Isotonic probability calibrator |
| `app.py` | FastAPI app — `/predict`, `/result`, `/health` |
| `fetch_data.py` | Download training data from martj42/international_results |
| `update_rankings.py` | Update FIFA rankings CSV with new values |
| `run_pipeline.py` | Shell-script-style full pipeline runner |
| `batch_predict.py` | Predict a batch of matches from CSV |

### Data files

| File | Contents |
|------|---------|
| `data/matches.csv` | Training data — historical international matches |
| `data/wc2022.csv` | WC 2022 fixture list with **actual** scores (64 matches) |
| `data/wc2026.csv` | WC 2026 fixture list — goals columns empty until played |
| `data/fifa_rankings_2022.csv` | FIFA rankings as of November 2022 (212 teams) |
| `data/fifa_rankings_2026.csv` | FIFA rankings as of April 1, 2026 (213 teams, official) |

### Model artefacts (generated, not committed)

`model.pkl`, `classifier.pkl`, `dc_ratings.pkl`, `calibrator.pkl`, `best_params.json`

---

## FIFA rankings — versioned system

`features.py` has two key functions:

```python
load_fifa_rankings(year)          # 2022 or 2026
set_fifa_rankings_year(year)      # sets global _FIFA_RANKINGS used for feature-building
```

Each script calls the correct year before importing predict/train:
- `evaluate.py` → called with `year=2022` or `year=2026` (auto-set inside `evaluate()`)
- `train.py` → `set_fifa_rankings_year(2026)`
- `simulate_wc2026.py` → `set_fifa_rankings_year(2026)`
- `main.py evaluate` → year=2022; `main.py evaluate2026` → year=2026

---

## Main CLI commands

```
python main.py train                             # train all 4 models (uses 2026 rankings)
python main.py tune [--trials N]                 # Optuna hyperparameter search
python main.py evaluate [--retrain] [--limit N]  # WC 2022 retrospective (2022 rankings)
python main.py evaluate2026 [--retrain] [--limit N]  # WC 2026 live (2026 rankings, skips empty rows)
python main.py simulate2026                      # full 2026 tournament simulation
python main.py predict "France" "Brazil"         # single match prediction
```

Standalone equivalents (all have `__main__` blocks):
```
python train.py
python evaluate.py [--year 2022|2026] [--retrain] [--limit N]
python simulate_wc2026.py
python tune.py [--trials N]
```

---

## Evaluate modes — frozen vs learning

| Mode | Behaviour | When to use |
|------|-----------|-------------|
| **Frozen** (default) | Model trained once before tournament, never retrained | Benchmarking, leaderboard, pre-tournament |
| **Learning** (`--retrain`) | Walk-forward: model retrains after each real result | Live tournament — model continuously improves |

Both modes available for both years.

---

## WC 2026 live tournament workflow (starts June 11, 2026)

1. Match is played → fill in `goals_A` / `goals_B` in `data/wc2026.csv`
2. `python main.py evaluate2026 --retrain` → walk-forward eval, exports `results_wc2026_retrain.xlsx`
3. `python main.py simulate2026` → re-simulates remaining bracket with improved model

If goals columns are empty, `evaluate2026` prints "No results available yet" and exits cleanly.

---

## Current prediction (as of April 7, 2026)

Model trained on `data/matches.csv` + Apr-2026 FIFA rankings.  
**Predicted WC 2026 champion: Mexico** (beat France 1-0 in final).  
Mexico's path: Switzerland (R16) → Brazil (QF) → Germany (SF) → France (Final).

---

## Planned future work

- **Automated live scoring**: fetch real-time scores → append to `wc2026.csv` → auto-retrain
- **Supabase**: store predictions/results via `app.py` FastAPI `/result` endpoint
- **Vercel frontend**: reads from Supabase, displays live bracket + predictions
- **Next FIFA rankings update**: June 10, 2026 (run `update_rankings.py` after updating the dict inside it)

---

## WC 2026 group structure

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

## April 2026 FIFA rankings (top 10)

1. France 1877.32  2. Spain 1876.40  3. Argentina 1874.81  4. England 1825.97
5. Portugal 1798.06  6. Brazil 1761.16  7. Netherlands 1759.32  8. Morocco 1755.87
9. Belgium 1734.71  10. Germany 1732.40
