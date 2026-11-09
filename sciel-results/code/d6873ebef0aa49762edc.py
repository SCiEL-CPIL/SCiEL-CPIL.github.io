"""Cost-aware acquisition for Hartmann-3D BO loop.
Combines EI/cost (current) with a lightweight calibration over a small
parameter grid (inspiration 1). Calibration selects (gamma, beta, w_ei)
that best ranks low-mu points highly among the top-k scored candidates.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # safety margin before 300s wall-clock limit


def _ei_score(mu, sigma, f_best, cost, beta=1.0):
    """Expected improvement per unit cost^beta."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    cost_safe = np.maximum(cost, 1e-9)
    return ei / (cost_safe ** beta)


def _ucb_score(mu, sigma, cost, gamma=1.0, beta=1.0):
    """UCB-like acquisition: -mu + gamma * sigma, cost-discounted."""
    cost_safe = np.maximum(cost, 1e-9)
    score = (-mu + gamma * sigma) / (cost_safe ** beta)
    return score


def _hybrid_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0, w_ei=0.5):
    """Hybrid of EI and UCB with weighted combination, normalized."""
    w_ucb = 1.0 - w_ei
    ei = _ei_score(mu, sigma, f_best, cost, beta=beta)
    ucb = _ucb_score(mu, sigma, cost, gamma=gamma, beta=beta)

    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std

    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std

    score = w_ei * ei_norm + w_ucb * ucb_norm
    return score


def _calibrate(mu, sigma, f_best, cost, time_budget=2.5):
    """Lightweight calibration over a small grid.
    Proxy: negative mean mu of top-k scored points (lower mu = better).
    """
    t0 = time.time()

    gamma_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    beta_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
    w_vals = [0.3, 0.5, 0.7]

    best_proxy = -np.inf
    best_params = (1.0, 1.0, 0.5)

    top_k = min(20, max(1, len(mu) // 10))

    for gamma in gamma_vals:
        for beta in beta_vals:
            for w_ei in w_vals:
                if time.time() - t0 > time_budget:
                    return best_params

                w_ucb = 1.0 - w_ei
                scores = _hybrid_score(mu, sigma, f_best, cost,
                                       gamma=gamma, beta=beta, w_ei=w_ei)

                if not np.all(np.isfinite(scores)):
                    continue

                top_idx = np.argsort(scores)[-top_k:]
                mean_mu_top = np.mean(mu[top_idx])
                proxy = -mean_mu_top

                if proxy > best_proxy:
                    best_proxy = proxy
                    best_params = (gamma, beta, w_ei)

    return best_params


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Hybrid Acquisition (EI + UCB) with lightweight calibration.

    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs

    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert and sanitize
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)

    mu = np.where(np.isfinite(mu), mu, f_best + 1.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Calibrate parameters (quick, ~2.5s max)
    if time.time() < _DEADLINE - 5.0:
        gamma, beta, w_ei = _calibrate(mu, sigma, f_best, cost, time_budget=2.5)
    else:
        gamma, beta, w_ei = 1.0, 1.0, 0.5

    # Compute hybrid score
    scores = _hybrid_score(mu, sigma, f_best, cost,
                           gamma=gamma, beta=beta, w_ei=w_ei)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
