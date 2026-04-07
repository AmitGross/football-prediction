# dc_ratings.py — Dixon-Coles MLE strength ratings with time decay
#
# Fits per-team attack (alpha) and defense (beta) parameters such that:
#   lambda_ij = exp(mu + alpha_i - beta_j)   (goals A scores vs B)
#   lambda_ji = exp(mu + alpha_j - beta_i)   (goals B scores vs A)
#
# Maximises log-likelihood weighted by time decay: w_t = exp(-XI * days_ago)
# Uses Dixon-Coles correction on low scores (same rho as poisson.py).
# Identifiability: soft constraint mean(alpha) = 0.

import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

DC_RATINGS_PATH = 'dc_ratings.pkl'
XI    = 0.0065   # decay per day  (half-weight at ~107 days ≈ 3.5 months)
RHO   = -0.13   # Dixon-Coles low-score correction (matches poisson.py)


def _neg_log_likelihood(params, n_teams, idx_A, idx_B, goals_A, goals_B, weights):
    alpha = params[:n_teams]
    beta  = params[n_teams: 2 * n_teams]
    mu    = params[2 * n_teams]

    lam_A = np.exp(mu + alpha[idx_A] - beta[idx_B])
    lam_B = np.exp(mu + alpha[idx_B] - beta[idx_A])
    lam_A = np.maximum(lam_A, 1e-10)
    lam_B = np.maximum(lam_B, 1e-10)

    # Vectorised Poisson log-PMF
    ll_A = goals_A * np.log(lam_A) - lam_A - gammaln(goals_A + 1)
    ll_B = goals_B * np.log(lam_B) - lam_B - gammaln(goals_B + 1)

    # Dixon-Coles low-score correction (vectorised)
    tau = np.ones(len(goals_A))
    m00 = (goals_A == 0) & (goals_B == 0)
    m10 = (goals_A == 1) & (goals_B == 0)
    m01 = (goals_A == 0) & (goals_B == 1)
    m11 = (goals_A == 1) & (goals_B == 1)
    tau[m00] = 1.0 - lam_A[m00] * lam_B[m00] * RHO
    tau[m10] = 1.0 + lam_B[m10] * RHO
    tau[m01] = 1.0 + lam_A[m01] * RHO
    tau[m11] = 1.0 - RHO
    tau = np.maximum(tau, 1e-10)

    nll = -(weights * (np.log(tau) + ll_A + ll_B)).sum()
    # Soft identifiability constraint: mean(alpha) ≈ 0
    nll += 500.0 * (alpha.mean() ** 2)
    return nll


def fit_dc_ratings(df, xi=XI, path=DC_RATINGS_PATH):
    """
    Fit time-decayed Dixon-Coles ratings on df and save to dc_ratings.pkl.
    Returns the ratings dict: {team: {'alpha': float, 'beta': float}, '_mu': float}
    """
    df = df.sort_values('date').reset_index(drop=True)
    ref_date  = df['date'].max()
    days_ago  = (ref_date - df['date']).dt.days.values.astype(float)
    weights   = np.exp(-xi * days_ago)

    teams     = sorted(set(df['team_A'].tolist() + df['team_B'].tolist()))
    team_idx  = {t: i for i, t in enumerate(teams)}
    n         = len(teams)

    idx_A   = np.array([team_idx[t] for t in df['team_A']], dtype=int)
    idx_B   = np.array([team_idx[t] for t in df['team_B']], dtype=int)
    goals_A = df['goals_A'].values.astype(float)
    goals_B = df['goals_B'].values.astype(float)

    # Initial params: alpha=0, beta=0, mu=log(avg goals per team)
    mu0 = np.log(max((goals_A.mean() + goals_B.mean()) / 2.0, 0.1))
    x0  = np.zeros(2 * n + 1)
    x0[2 * n] = mu0

    print(f"Fitting DC ratings: {n} teams, {len(df)} matches ...")
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(n, idx_A, idx_B, goals_A, goals_B, weights),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6},
    )
    print(f"  converged={result.success}  nll={result.fun:.2f}")

    alpha = result.x[:n] - result.x[:n].mean()   # enforce zero-mean
    beta  = result.x[n: 2 * n]
    mu    = result.x[2 * n]

    ratings = {'_mu': float(mu)}
    for i, t in enumerate(teams):
        ratings[t] = {'alpha': float(alpha[i]), 'beta': float(beta[i])}

    with open(path, 'wb') as f:
        pickle.dump(ratings, f)
    print(f"DC ratings saved to {path}")
    return ratings


def predict_dc_lambdas(team_A, team_B, ratings):
    """
    Returns (lam_A, lam_B) from DC strength ratings.
    Falls back to (None, None) if ratings is None.
    Unknown teams get alpha=0, beta=0 (global average).
    """
    if ratings is None:
        return None, None
    mu  = ratings.get('_mu', np.log(1.3))
    r_A = ratings.get(team_A, {'alpha': 0.0, 'beta': 0.0})
    r_B = ratings.get(team_B, {'alpha': 0.0, 'beta': 0.0})
    lam_A = float(np.exp(mu + r_A['alpha'] - r_B['beta']))
    lam_B = float(np.exp(mu + r_B['alpha'] - r_A['beta']))
    return max(lam_A, 1e-4), max(lam_B, 1e-4)


def load_dc_ratings(path=DC_RATINGS_PATH):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    df = pd.read_csv('data/matches.csv', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    fit_dc_ratings(df)
