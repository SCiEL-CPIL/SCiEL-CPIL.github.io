import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization on Shekel 4D.
    Uses Lower Confidence Bound with adaptive exploration parameter.
    
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
        # Safety: return exploit-heavy fallback
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Lower Confidence Bound for minimization:
    # We want to minimize, so LCB = mu - beta * sigma
    # We maximize the negative: score = -(mu - beta * sigma) / cost = (-mu + beta * sigma) / cost
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # Use a smooth decay based on call count
    if _call_count <= 5:
        beta = 2.0
    elif _call_count <= 10:
        beta = 1.5
    elif _call_count <= 20:
        beta = 1.0
    elif _call_count <= 40:
        beta = 0.5
    else:
        beta = 0.3
    
    # LCB score: higher is better
    lcb_score = (-mu + beta * sigma) / cost_safe
    
    # Add a small PI (Probability of Improvement) component to break ties
    # and provide additional signal near the best
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    pi_cost_aware = pi / cost_safe
    
    # Combine: primarily LCB, with a small PI boost for points likely to improve
    # The PI term helps when LCB scores are similar
    scores = lcb_score + 0.1 * pi_cost_aware
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
