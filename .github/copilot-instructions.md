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
- **Features (62 total, v1.5)**: Elo ratings, form (last 5 + last 2), H2H, days rest, FIFA rankings, neighbourhood aggregation + performance-aware neighbourhood features (weighted_opp_elo, win_rate_vs_top_teams, avg_goal_diff_vs_opp, weighted_goal_diff_by_opp). Kalman filter (x10) and PageRank/HITS (x12) removed in v1.5 after ablation showed they hurt performance.
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

## Git branch state (as of April 9, 2026)

| Branch | Version | Features | Status |
|--------|---------|----------|--------|
| `main` | v1.4 | 84 | Stable — Netherlands champion simulation |
| `v1.5-dev` | v1.5 | 62 | **Ready to merge** — Spain champion simulation |
| `v1.6-dev` | v1.6 | 67 (planned) | **Next** — create off main after v1.5 merge |

**Current active branch**: `v1.5-dev`  
Next action: merge `v1.5-dev` → `main`, then create `v1.6-dev`.

---

## Current prediction (as of April 9, 2026 — model v1.5)

Model trained on `data/matches.csv` + Apr-2026 FIFA rankings (62 features).  
**Predicted WC 2026 champion: Spain** (beats France 2-1 in Final).  
Spain's path: Egypt (R32) → Belgium (R16) → Germany (QF) → France (SF) → Final.

**v1.5 WC 2022 benchmarks (frozen):** Accuracy 50.0% · RPS 0.2147 · RMSE 1.369  
**v1.5 WC 2022 benchmarks (retrain):** Accuracy 53.1% · RPS 0.2088 · RMSE 1.3199  
**v1.5 WC 2018 benchmarks (frozen):** Accuracy 42.2% · RPS 0.2549 · RMSE 1.299  
**v1.5 WC 2018 benchmarks (retrain):** Accuracy 39.1% · RPS 0.2490 · RMSE 1.2151

---

## Ablation study results (WC 2022, v1.4 baseline)

Ran full ablation leaving out one feature group at a time. dRPS = change in RPS when group removed (positive = group helps, removing it hurts).

### Full tournament frozen (64 matches) — Baseline: Acc=48.4%, RPS=0.2236

| Group | Features | dRPS | Verdict |
|-------|----------|------|---------|
| FIFA Rankings | 3 | +0.0169 | **Most valuable** |
| Rest & Match Count | 4 | +0.0018 | Small positive |
| PageRank/HITS | 12 | **-0.0138** | **Most harmful → REMOVED** |
| Neighbourhood Basic | 9 | -0.0050 | Hurts full, critical in KO |
| Kalman | 10 | -0.0005 | Mildly hurts → REMOVED |

### Knockout frozen (16 matches) — Baseline: Acc=62.5%, RPS=0.1256

| Group | Features | dRPS | Verdict |
|-------|----------|------|---------|
| FIFA Rankings | 3 | +0.0291 | **Dominant** |
| Neighbourhood Basic | 9 | +0.0134 | Critical in KO |
| Elo | 3 | +0.0066 | Valuable |
| PageRank/HITS | 12 | -0.0058 | Harmful → REMOVED |
| Kalman | 10 | -0.0019 | Hurts in KO → REMOVED |

**Decision**: Removed Kalman (x10) + PageRank/HITS (x12) → 84 features → 62 features (v1.5)

---

## v1.5 feature set (62 features)

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

---

## Planned v1.6 features (5 new → 67 total)

To be implemented on `v1.6-dev` branch after v1.5 merges to main:

| Feature | Type | Values | Rationale |
|---------|------|--------|-----------|
| `is_knockout` | binary | 0/1 | Group vs KO context |
| `round_number` | ordinal rank | 0=qualifier, 1=group, 2=R32/R16, 3=QF, 4=SF, 5=Final | Model learns behavior changes as tournament progresses |
| `games_played_in_tournament` | count | 0,1,2,3... | Momentum/fatigue accumulation |
| `goal_diff_std_A` | float | std dev last 5 GF-GA | Team volatility/consistency |
| `goal_diff_std_B` | float | std dev last 5 GF-GA | Team volatility/consistency |

Non-tournament matches default to 0 for all stage features.

### Why these 5 — full decision rationale

Features considered and rejected for v1.6:

- **Recent Form (3-match)** — already covered by Form-2 and Form-5 in v1.5 (14+14 features including per-team weighted form). Adding a Form-3 variant unlikely to add signal on top of existing form features.
- **Strength of Schedule (SoS)** — already covered by v1.5 neighbourhood features: `weighted_opp_elo`, `avg_opp_elo`, `win_rate_vs_top_teams`, `avg_goal_diff_vs_opp`. Adding an explicit `sos_diff` would be partly redundant.
- **Kalman filter** — do NOT bring back. Was ablated out in v1.5 (hurt performance in both full-tournament and KO evaluation). Note: it already had explicit attack/defense separation (`ka_atk`, `ka_def`, `kb_atk`, `kb_def`, `ka_unc_atk`, `ka_unc_def` + diffs — 10 features) and was still harmful, so that is not a reason to revisit it.

What IS missing: the model currently treats group match 1 and the Final identically. Stage features fix this.

### v1.6 incremental ablation plan

Do NOT add all 5 features at once. Test in order:

1. **Baseline** — v1.5 as-is (62 features)
2. **+stage** — add `is_knockout`, `round_number`, `games_played_in_tournament` (→ 65 features)
3. **+stage +volatility** — add `goal_diff_std_A`, `goal_diff_std_B` (→ 67 features)

Evaluate each step on WC 2022 frozen + retrain. Only keep a step if RPS improves. Focus evaluation on group stage (most room for improvement) and knockout (where stage signal should matter most).

---

## Bug fixes (v1.5)

- `train.py` `save_model()`: Fixed Windows `OSError: [Errno 22] Invalid argument: 'model.pkl'` during retrain mode — now writes to `.tmp` file then uses `os.replace()` (atomic swap) instead of direct `open(path, 'wb')`

---

## Planned future work

- **v1.6**: Tournament stage features + volatility (5 features, see above)
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
