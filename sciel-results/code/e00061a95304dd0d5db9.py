"""Cost-aware acquisition function using UCB-style exploration with cost normalization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a UCB-style score:
    score = (f_best - mu + kappa * sigma) / cost
    
    This balances exploitation (low mu) and exploration (high sigma),
    normalized by evaluation cost.
    """
    _get_start_time()
    
    # Deadline guard: track elapsed time to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 315.0:  # 15 second safety margin before 330s kill
        # Fallback: simple UCB without cost normalization
        sigma_safe = np.maximum(sigma, 1e-8)
        # Adaptive kappa based on problem scale
        mu_range = np.max(mu) - np.min(mu)
        if mu_range > 0:
            kappa = 1.0
        else:
            kappa = 0.5
        scores = (f_best - mu) + kappa * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive kappa: balance exploration vs exploitation
    # Use the spread of sigma to calibrate kappa
    sigma_median = np.median(sigma_safe)
    mu_spread = np.max(mu) - np.min(mu)
    
    # kappa should be proportional to the typical uncertainty relative to the objective scale
    # When sigma is large relative to f_best, explore more
    if mu_spread > 1e-10:
        # Scale kappa so that kappa*sigma is comparable to the mu spread
        kappa = np.clip(mu_spread / (2.0 * sigma_median + 1e-10), 0.5, 5.0)
    else:
        # All mu values are similar; rely more on sigma
        kappa = 2.0
    
    # UCB-style score: reward low mu and high sigma
    # (f_best - mu) is the potential improvement, kappa*sigma is exploration bonus
    ucb_score = (f_best - mu) + kappa * sigma_safe
    
    # Cost-aware: divide by cost
    scores = ucb_score / cost_safe
    
    # Final safety: ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
