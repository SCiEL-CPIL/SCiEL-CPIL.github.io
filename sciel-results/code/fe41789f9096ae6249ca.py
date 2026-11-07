"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time across calls
_start_time = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Lower Confidence Bound (LCB) for exploitation/exploration balance,
    combined with linear cost penalty to maximize evaluations within budget.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    global _start_time, _DEADLINE
    
    # Check if we've exceeded the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple fallback: prefer low cost
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-6)
        return 1.0 / cost
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero or negative values
    cost = np.maximum(cost, 1e-6)
    
    # LCB: mu - beta * sigma (lower is better, so we negate for maximization)
    # beta controls exploration-exploitation tradeoff
    # Use a moderate beta to balance exploration and exploitation
    beta = 2.0
    
    # LCB score (lower is better, so we negate for maximization)
    lcb = -(mu - beta * sigma)
    
    # Cost-aware scaling: use linear cost penalty
    # This allows more evaluations within the fixed budget
    # Linear penalty is less aggressive than quadratic, allowing better exploration
    cost_factor = 1.0 / cost
    
    # Combine LCB and cost-aware scaling
    score = lcb * cost_factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
