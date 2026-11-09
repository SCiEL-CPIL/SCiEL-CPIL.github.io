"""Cost-aware acquisition: UCB-per-cost as the primary signal."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if close to limit, return a fast fallback
    if not _elapsed_ok():
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0) / (cost + 1e-9)

    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (UCB) for minimization:
    # We want to maximize the score, so we use:
    # UCB = -mu + beta * sigma
    # But since we're minimizing, lower mu is better.
    # A common formulation for minimization is:
    # acquisition = -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Adaptive beta: scale with the standard deviation of mu
    mu_std = np.std(mu)
    if mu_std < 1e-12:
        mu_std = 1e-12
        
    # beta controls the exploration-exploitation tradeoff
    # A typical value is around 2.0, but we can adapt it
    beta = 2.0
    
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB: use a weighted sum
    # Normalize both to have comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 1e-12 else 1.0
    
    # Weight: 0.6 for EI, 0.4 for UCB
    weighted_score = 0.6 * (ei / ei_max) + 0.4 * (ucb / ucb_max)
    
    # Cost-aware: divide by cost (or subtract a cost penalty)
    # Dividing by cost penalizes expensive evaluations more strongly
    score = weighted_score / (cost + 1e-9)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
