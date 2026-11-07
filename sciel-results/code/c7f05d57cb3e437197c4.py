import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        # Simple exploitation: prefer low mu and low cost
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Structural Change: Switch from Expected Improvement (EI) to Upper Confidence Bound (UCB)
    # UCB for minimization: We seek to minimize the optimistic lower bound: mu - beta * sigma.
    # The score (higher is better) is the negative of this bound, normalized by cost.
    # Score = (beta * sigma - mu) / cost
    
    # Beta controls the exploration/exploitation trade-off.
    # A value of 2.0 is a common heuristic for BO.
    beta = 2.0
    
    # Calculate UCB score
    # Note: Unlike EI, UCB can be negative if mu is very high and sigma is low.
    # This allows the algorithm to strongly avoid bad, expensive points.
    ucb_score = (beta * sigma_safe - mu) / safe_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    return score
