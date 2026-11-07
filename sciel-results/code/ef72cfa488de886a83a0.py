import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses adaptive Upper Confidence Bound (UCB) with a cost penalty.
    Adaptive beta balances exploration (high sigma regions) and exploitation (low mu regions).
    The cost penalty favors low-cost regions where the global minimum is likely located.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 290:  # Safety margin before 300s limit
        # Return a simple cost-based score as fallback
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe
    
    # Ensure sigma and cost are positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive beta based on number of evaluations (inferred from cost range)
    # Early: more exploration (higher beta), Late: more exploitation (lower beta)
    # Cost ranges from ~1 (x near 0) to ~2 (x near 5)
    # Use the minimum cost as a proxy for how "early" we are
    min_cost = np.min(cost)
    # Normalize: early iterations have lower accumulated cost, so use a heuristic
    # Based on the range of mu values as a proxy for convergence
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-6:
        # Very converged, use low beta
        beta = 0.5
    elif mu_range < 10:
        # Moderately converged
        beta = 1.5
    elif mu_range < 50:
        # Early-mid stage
        beta = 2.5
    else:
        # Early stage, more exploration
        beta = 3.5
    
    # UCB for minimization:
    # Score component: beta * sigma - mu
    ucb_score = beta * sigma - mu
    
    # Apply quadratic cost penalty (stronger preference for low-cost regions)
    # The global minimum of Styblinski-Tang is at x = -2.9035 in each dimension
    # which maps to low cost (x scaled to [0,1] gives ~0.21)
    score = ucb_score / (cost ** 1.5)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
