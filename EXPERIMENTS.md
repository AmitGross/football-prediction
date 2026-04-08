# Football Prediction Project — Experiments & Iteration Log

> **Current production model: v1.1** (84 features, tagged `v1.1` on GitHub)  
> To reproduce: `git checkout v1.1` → `python main.py train` → `python main.py evaluate`

## Version Summary

| Version | Features | Retrain Accuracy | Retrain RPS | Retrain RMSE | KO Accuracy | KO RPS | Status |
|---------|----------|-----------------|-------------|--------------|-------------|--------|--------|
| v1.0 | 72 | 50.0% | 0.2119 | 1.4142 | 75.0% | 0.1364 | superseded |
| **v1.1** | **84** | **53.1%** | **0.2094** | **1.3607** | **75.0%** | **0.1333** | ✅ **current** |
| v1.2 | 93 | 51.6% | 0.2059 | 1.3110 | 68.8% | 0.1330 | reverted |
| v1.3 | 75 | TBD (reverted early) | — | — | — | — | reverted |

---

## Iteration Details

### v1.0 — Baseline (72 features)
- Elo, Kalman, form (last 5 + last 2), H2H, days rest, FIFA rankings, PageRank, HITS, neighbourhood Elo aggregation
- WC 2022 retrain: 50.0% accuracy, RPS 0.2119, RMSE 1.4142
- Tagged `v1.0-baseline` on GitHub

### v1.1 — Performance-aware neighbourhood features (84 features) ✅ CURRENT
**What changed:** Extended `calculate_neighbourhood_features()` with 4 outcome-conditional features:
- `weighted_opp_elo` — opponent Elo × outcome (+1 win / 0 draw / −1 loss)
- `win_rate_vs_top_teams` — win rate vs top 30% Elo opponents
- `avg_goal_diff_vs_opp` — average GF−GA across all past opponents
- `weighted_goal_diff_by_opp` — goal margin scaled by opponent Elo

**Results (WC 2022):**
- Retrain: 53.1% accuracy (+3.1pp), RPS 0.2094 (↓), RMSE 1.3607 (↓) — all better than v1.0
- Knockout (16 matches): 75% accuracy, RPS 0.1333, RMSE 1.199
- WC 2026 simulation: **Mexico wins** (beat France 1-0 in final)

**Tagged `v1.1` on GitHub — this is the trusted production version.**

### v1.2 — Transitive win-graph features (93 features) — REVERTED
**What changed:** Added `calculate_wingraph_features()` — 2-hop outcome-conditional win graph:
- `hop1_win_score`: Σ(win_margin × opp_Elo) / n_matches
- `hop2_win_score`: hop1 of beaten opponents × 0.5 decay (chain dies on draws/losses)
- `transitive_dominance`: hop1 + hop2

**Results vs v1.1:**
- RMSE and RPS improved slightly (noise reduction in score prediction)
- Knockout accuracy dropped: 68.8% vs 75% (1 extra match wrong)

**Why reverted:** hop1 is redundant with `weighted_goal_diff_by_opp` already in v1.1. hop2 is mostly zeros in walk-forward context (sparse win chains early in tournament). Not worth the accuracy trade-off.

### v1.3 — Drop HITS/goal_pr (75 features) — REVERTED
**What changed:** Removed `goal_pr`, `hub`, `auth` HITS features — kept only `win_pr`

**Rationale tested:** HITS/goal_pr are built on goal graph regardless of win/loss, so fixture-draw randomness pollutes the signal. Only `win_pr` (win graph, outcome-conditional) is theoretically sound.

**Results:** Frozen eval already worse (42.2% accuracy vs 45.3%, RPS 0.2182 vs 0.2160). Reverted before retrain eval completed.

**Why reverted:** Despite the theoretical argument, removing these features hurt empirically. The features may capture some noise but also genuine signal (defensive vulnerability, offensive volume) that the model uses. Leaving them in.

---

## 1. Baseline Model
- **Model:** RandomForestRegressor + XGBoostRegressor (AveragingEnsemble)
- **Features:** 72 (Elo, Kalman, form, head-to-head, PageRank, HITS, temporal decay, 1-hop aggregation, FIFA ranking, rest, etc.)
- **Target:** Predict goals for team_A and team_B (regression)
- **Output:** Poisson probability grid (Dixon-Coles correction)
- **Score display:** Amplified λ (power transform, α=1.5) for realistic scorelines
- **Best hyperparameters:**
  - rf_n_estimators: 368
  - rf_max_depth: 12
  - xgb_n_estimators: 313
  - xgb_max_depth: 8
  - xgb_learning_rate: 0.059
  - xgb_subsample: 0.768
  - xgb_colsample_bytree: 0.627
  - elo_k: 25.5
  - process_noise: 11.6
  - measure_noise: 4.6
- **Results (WC 2022 test set, 64 matches):**
  - RMSE combined: 1.417
  - Outcome accuracy: 48.4%
  - **Mean RPS: 0.2186**

---

## 2. Graph Features & Temporal Decay
- **Added:**
  - PageRank (win/goal graphs, temporal decay)
  - HITS (hub/authority, temporal decay)
  - 1-hop neighbourhood aggregation (GNN-style)
- **Result:**
  - No significant RPS improvement (remained ~0.218–0.220)

---

## 3. Lambda Amplification
- **Purpose:** Make predicted scores more realistic (not always 1-0/1-1)
- **Method:** Power transform λ → λ^α (α=1.5) for display only
- **Result:**
  - RMSE improved (1.536 → 1.417)
  - RPS unchanged (0.2200 → 0.2186)

---

## 4. Advanced Blending (DC, Classifier, Calibration)
- **Tried:**
  - Dixon-Coles MLE strength ratings (dc_ratings.py)
  - XGBClassifier (W/D/L outcome)
  - Isotonic regression calibration
  - Blending all signals (regressor λ, DC λ, classifier, calibrator)
- **Result:**
  - RPS worsened (0.2186 → 0.2588)
  - Reason: DC ratings did not converge (too little data/team), classifier/calibrator overfit
  - Disabled all advanced blending for now

---

## 5. Recommendations & Next Steps
- **Best model:** Baseline Poisson probabilities from regressor λ only (no DC, no classifier, no calibrator)
- **To improve further:**
  - Add more training data (qualifiers, friendlies, Nations League)
  - Re-enable DC, classifier, calibration with proper regularization and held-out calibration set
  - Target RPS < 0.21 (current: 0.2186)

---

## 6. How to Run
- `python main.py train` — retrain all models (regressor, classifier, DC, calibrator)
- `python main.py evaluate` — evaluate on WC 2022 test set (uses best model by default)
- `python main.py predict "France" "Brazil"` — predict a single match

---

## 7. Terminology
- **λ (lambda):** Expected goals for a team, predicted by the model.
- **Poisson model:** Uses λ to generate a probability grid for all possible scorelines.
- **Dixon-Coles correction:** Adjustment to Poisson for low scores (0-0, 1-0, 0-1, 1-1).
- **PageRank:** Graph-based feature measuring team dominance (win/goal graphs).
- **HITS (hub/authority):** Graph features for attack/defense quality.
- **Kalman filter:** Tracks team attack/defense strength over time.
- **Elo rating:** Classic rating system for team strength, updated after each match.
- **Form features:** Recent win/draw/loss and goals scored/conceded.
- **Neighbourhood aggregation:** GNN-style feature: aggregates opponent stats.
- **Amplified λ:** Power transform to λ for more realistic scorelines (display only).
- **Frozen model:** Model trained on all pre-tournament data, evaluated on test set without retraining.
- **Walk-forward:** Features and predictions use only past data up to each match (no leakage).
- **RPS (Ranked Probability Score):** Measures probability calibration for ordered outcomes (W/D/L). Lower is better.
- **RMSE (Root Mean Squared Error):** Measures error in predicted goals.
- **Outcome accuracy:** Fraction of matches where predicted outcome (W/D/L) matches actual.

---

## 8. Metrics & Loss Functions
- **RPS (Ranked Probability Score):**
  - $\text{RPS} = \frac{1}{2} \sum_{k=1}^{K-1} (F_k - O_k)^2$
  - $F_k$ = cumulative predicted probability up to outcome $k$
  - $O_k$ = cumulative actual outcome (1 for true, 0 for false)
  - Lower is better. Perfect = 0. Typical good: <0.21
- **RMSE (Root Mean Squared Error):**
  - $\sqrt{\frac{1}{N} \sum (\hat{y} - y)^2}$
  - Measures error in predicted goals
- **Outcome accuracy:**
  - Fraction of matches where predicted W/D/L matches actual

---

## 9. Workflow & Lessons
- Feature engineering: walk-forward, no leakage
- Model selection: Optuna hyperparameter search
- Evaluation: frozen model on WC 2022 test set
- All experiments, metrics, and results tracked in this file
- **Key lesson:** Probability calibration and data quantity are more important than model complexity for RPS

---

## 10. Session Summary
- Explored many advanced features and blending methods
- Best RPS achieved with baseline Poisson model (regressor λ only)
- Advanced blending (DC, classifier, calibration) hurt RPS due to overfitting/data sparsity
- Next step: add more data, then revisit advanced methods

---

_Last updated: April 7, 2026_
