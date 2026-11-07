"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid LCB/EI strategy with adaptive exploration.
    
    For Rastrigin (multimodal), aggressive exploration of high-variance regions
    is crucial. We use a Lower Confidence Bound (LCB) with an adaptive beta that
    starts high (exploration) and decays as total cost increases (exploitation).
    
    The score is the optimistic improvement (f_best - LCB) normalized by cost.
    """
    # Deadline check: if we're close to the limit, return a safe fallback
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive exploration coefficient beta
    # Mean cost reflects the "budget spent" so far.
    # Start with high exploration (beta ~ 2.0) and decay to low (beta ~ 0.5).
    # Cost ranges from ~5 (start) to 30 (end).
    # Linear decay: beta = 2.5 - 0.1 * mean_cost
    # At cost=5: beta = 2.0
    # At cost=30: beta = 0.5
    mean_cost = np.mean(cost)
    beta = 2.5 - 0.1 * mean_cost
    beta = np.clip(beta, 0.5, 3.0)
    
    # Lower Confidence Bound (for minimization)
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Potential improvement: how much better could the LCB be than f_best?
    # We want to maximize this, but only if it's positive (i.e., LCB < f_best)
    improvement = f_best - lcb
    
    # Ensure improvement is non-negative (if LCB > f_best, improvement is negative, clamp to 0)
    improvement = np.maximum(improvement, 0.0)
    
    # Cost-aware score: improvement per unit cost
    score = improvement / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
