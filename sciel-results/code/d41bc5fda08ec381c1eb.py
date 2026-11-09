import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) with log-cost penalty.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return np.ones_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is valid for log
    cost_safe = np.maximum(cost, 1e-6)
    
    # LCB parameters
    kappa = 2.5
    alpha = 5.0
    
    # LCB for minimization: we want to maximize (kappa * sigma - mu)
    lcb_score = kappa * sigma - mu
    
    # Log-cost penalty: penalize high cost points
    # log(cost) is 0 for cost=1, ~0.69 for cost=2
    # We subtract this to make high cost points less attractive
    penalty = alpha * np.log(cost_safe)
    
    scores = lcb_score - penalty
    
    # Handle non-finite values
    # If sigma is huge, lcb_score is huge (good).
    # If mu is -inf, lcb_score is +inf.
    # If mu is +inf, lcb_score is -inf.
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
