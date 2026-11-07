"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Design:
    - Uses a dynamic exploration parameter (kappa) that decays as f_best approaches 0.
      Early on (high f_best), we explore more aggressively (high kappa).
      Late on (low f_best), we exploit more (low kappa).
    - Uses a multiplicative cost penalty: score = (f_best - lcb) * exp(-lambda * cost).
      This naturally penalizes expensive points without needing additive scaling.
    - LCB is computed as mu - kappa * sigma.
    
    The key insight is that the optimal balance between exploration and exploitation
    depends on how close we are to the optimum. Since the minimum is 0, we can use
    f_best as a proxy for progress.
    """
    # Deadline Guard
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple LCB with fixed kappa
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        lcb = mu - 2.0 * sigma_safe
        score = -lcb
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Dynamic kappa: start high, decay as f_best approaches 0
    # f_best is typically in [0, 100] initially, decreasing to ~0
    # Use a smooth decay: kappa = 1.0 + 3.0 / (1.0 + f_best)
    # When f_best is large (e.g., 100), kappa ~ 1.0 + 3/101 ~ 1.03
    # When f_best is small (e.g., 0.1), kappa ~ 1.0 + 3/1.1 ~ 3.73
    # This gives more exploration when we're far from optimum
    
    f_best_safe = max(f_best, 1e-6)
    kappa = 1.0 + 3.0 / (1.0 + f_best_safe)
    
    # Compute LCB (Lower Confidence Bound) for minimization
    lcb = mu - kappa * sigma_safe
    
    # Potential improvement relative to best
    improvement = f_best - lcb
    
    # Cost-aware penalty: exponential decay with cost
    # cost is in [1, 2]
    # Use lambda = 0.5 for moderate penalty
    lambda_cost = 0.5
    cost_penalty = np.exp(-lambda_cost * cost)
    
    # Score: higher is better
    # We want high improvement and low cost
    score = improvement * cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
