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
    if elapsed > 315:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    sigma = np.clip(sigma, 1e-8, 1e4)
    cost_safe = np.maximum(cost, 1e-9)

    # z-score for minimization EI
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    cai = ei / cost_safe

    # Adaptive UCB exploration: stronger early, decays over time
    # Total budget is 30 cost units; we have roughly 15-20 evaluations
    # so _call_count ranges from 1 to ~25
    if _call_count <= 3:
        beta = 0.15
    elif _call_count <= 8:
        beta = 0.08
    elif _call_count <= 15:
        beta = 0.03
    else:
        beta = 0.01

    # UCB term: explore where uncertainty is high, penalize by cost
    ucb = beta * sigma / cost_safe

    scores = cai + ucb

    # Also add a small exploitation term for points already better than f_best
    # This helps refine near known good points
    exploit = np.maximum(f_best - mu, 0.0) / cost_safe * 0.5
    scores = scores + exploit

    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
