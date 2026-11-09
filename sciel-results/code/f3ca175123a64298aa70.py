import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    cost_safe = np.maximum(cost, 1e-9)
    cost_aware_ei = ei / cost_safe
    
    # Adaptive exploration bonus: stronger early, weaker later
    # In early iterations, explore more; in later iterations, exploit more
    if _call_count <= 5:
        exploration_factor = 0.10
    elif _call_count <= 10:
        exploration_factor = 0.07
    elif _call_count <= 20:
        exploration_factor = 0.04
    else:
        exploration_factor = 0.02
    
    exploration_bonus = sigma / cost_safe * exploration_factor
    
    # Combine
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
