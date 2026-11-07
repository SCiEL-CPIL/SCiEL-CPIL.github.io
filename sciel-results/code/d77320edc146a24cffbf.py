"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe default
    elapsed = _get_elapsed()
    if elapsed > 320:
        return np.zeros(len(mu), dtype=np.float64)

    # Cost-aware UCB acquisition with global budget-aware scaling
    # Use UCB: score = (f_best - mu) + beta * sigma
    # Then apply a global cost penalty based on relative cost within the batch
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # UCB score: optimistic estimate of improvement
    # beta controls exploration vs exploitation
    # Use a moderate beta for aggressive exploration
    beta = 2.0
    ucb = (f_best - mu) + beta * sigma_clipped
    
    # Global budget-aware scaling:
    # Compute mean cost of the candidate batch
    mean_cost = np.mean(cost)
    cost_std = np.std(cost)
    
    # Normalize cost relative to batch mean
    # Points with higher-than-average cost get penalized more
    # Points with lower-than-average cost get less penalty
    if cost_std > 1e-10:
        normalized_cost = (cost - mean_cost) / cost_std
    else:
        normalized_cost = np.zeros_like(cost)
    
    # Apply a penalty proportional to the normalized cost
    # This shifts the exploration-exploitation balance based on remaining budget
    # Higher cost relative to batch mean -> larger penalty
    penalty_weight = 0.5
    scores = ucb - penalty_weight * normalized_cost * np.mean(np.abs(ucb))
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
