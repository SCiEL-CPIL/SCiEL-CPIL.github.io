import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    """
    global _start_time

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return simple PMM to avoid timeout
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)

    # For sphere 1d on [-5, 5], f(x) = x^2, so f >= 0 and max is 25
    # Progress: 0 when f_best is large (early), 1 when f_best is near 0 (late)
    # Use a smooth transition
    max_possible = 25.0
    progress = 1.0 - np.clip(f_best / max_possible, 0.0, 1.0)

    # Compute PMM score (exploitation): -mu
    pmm_score = -mu

    # Compute UCB score (exploration): mu - kappa * sigma
    # For minimization, lower UCB is better, so we want higher score for lower UCB
    # UCB = mu - kappa * sigma, so -UCB = -mu + kappa * sigma
    # We want higher score to mean "more promising", so use -UCB
    kappa = 1.0
    ucb_score = -mu + kappa * sigma

    # Normalize both to comparable scales
    # PMM is in objective units, UCB is also in objective units
    # Let's normalize them to [0, 1] range for blending
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    pmm_norm = normalize(pmm_score)
    ucb_norm = normalize(ucb_score)

    # Adaptive weights: early -> more UCB (exploration), late -> more PMM (exploitation)
    # Use a smooth sigmoid-like transition
    # When progress is low (early), w_pmm is low, w_ucb is high
    # When progress is high (late), w_pmm is high, w_ucb is low
    transition_center = 0.5
    transition_width = 0.3
    t = (progress - transition_center) / transition_width
    w_pmm = 1.0 / (1.0 + np.exp(-t))  # sigmoid: 0 when t is very negative, 1 when very positive
    w_ucb = 1.0 - w_pmm

    # Ensure weights are in valid range
    w_pmm = np.clip(w_pmm, 0.1, 0.9)  # Always keep some exploration
    w_ucb = 1.0 - w_pmm

    # Blend the normalized scores
    score = w_pmm * pmm_norm + w_ucb * ucb_norm

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
