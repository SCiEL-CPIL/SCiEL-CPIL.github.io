"""
Structural rewrite: Cost-Weighted GP-UCB for Powell 4D.

Changes from previous version:
- Removed Expected Improvement (EI) and norm.cdf/pdf calculations.
- Removed manual min-max normalization of scores.
- Uses pure UCB: score = (-mu + beta * sigma) / cost^alpha
- This is structurally distinct: no normalization loops, no probability density functions.
- Fallback is also UCB-based for consistency.
"""
import numpy as np
import time

# Global state for deadline guard
_start_time = [None]
_DEADLINE = 290.0  # 300s limit with 10s safety margin

def _get_elapsed():
    """Return seconds since the first call to acquisition."""
    if _start_time[0] is None:
        _start_time[0] = time.time()
    return time.time() - _start_time[0]

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: array of shape (N,) predicted means.
        sigma: array of shape (N,) standard deviations.
        f_best: float, lowest observed value.
        cost: array of shape (N,) evaluation costs.
    
    Returns:
        array of shape (N,) scores. Higher is better.
    """
    elapsed = _get_elapsed()
    
    # Ensure numerical stability
    # Sigma can be zero or very small; floor it to avoid division issues if we were doing z-scores, 
    # but for UCB we just need finite inputs.
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # If we are running out of time, return a simple, fast score to prevent timeout
    if elapsed > _DEADLINE:
        # Fallback: Just use -mu / cost to get the best prediction as fast as possible
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    # UCB for minimization:
    # We want to minimize the function.
    # High score = likely better (lower) function value.
    # Predicted value is ~ N(mu, sigma^2).
    # We prefer low mu and high sigma (uncertainty might hide a lower minimum).
    # UCB_min = -mu + beta * sigma
    
    # Tuning parameter:
    # beta = 2.0 is a strong exploration weight.
    # For Powell 4D, which has a complex landscape, high exploration is often beneficial
    # to find the global basin, especially since the "cost" encourages staying low
    # (where the function might be flatter or higher), so we need the UCB boost to
    # push us into unexplored areas even if mu is not the absolute lowest.
    
    beta = 2.0
    alpha = 1.0
    
    # Calculate raw UCB score
    ucb_score = -mu + beta * sigma_safe
    
    # Apply cost-awareness.
    # Dividing by cost^alpha favors candidates with lower cost.
    # Cost is 1 + mean(x_scaled).
    # Lower x -> lower cost -> higher score (assuming positive ucb_score).
    # If ucb_score is negative (unlikely if we found a good f_best, but possible early on),
    # dividing by cost will make it *less* negative (closer to 0), which is correct
    # if we consider 0 as a neutral score and we want to prioritize cost when value is poor?
    # Actually, if ucb_score is negative, -1/2 > -1/1. So lower cost makes the score "better" (less negative).
    # This aligns with preferring cheaper points when the predicted value is bad.
    
    score = ucb_score / (cost_safe ** alpha)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
