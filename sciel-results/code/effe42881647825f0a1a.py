"""Cost-aware acquisition using adaptive UCB with cost normalization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s wall-clock safety margin (limit is 330s)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive beta: balance exploration based on current sigma scale
    # Use median sigma to adapt the exploration weight
    sigma_median = np.median(sigma_safe)
    
    # Adaptive beta: higher when sigma is large relative to f_best scale
    # This provides stronger exploration when uncertainty is high
    beta = 2.0 * np.sqrt(np.log(1 + sigma_median + 1e-8))
    
    # UCB: mu + beta * sigma (for minimization, we want low mu and high sigma)
    # Score = (f_best - mu) + beta * sigma, normalized by cost
    ucb_score = (f_best - mu) + beta * sigma_safe
    
    # Cost-aware: divide by cost to get improvement per unit cost
    score = ucb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
