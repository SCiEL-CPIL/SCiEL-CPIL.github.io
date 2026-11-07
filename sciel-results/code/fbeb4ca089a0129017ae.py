"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to enforce deadline
_start_time = None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted LCB acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 290:  # Safety margin before 300s limit
        # Return scores based on simple negative mean to finish quickly
        return -mu.astype(np.float64)
    
    # Ensure inputs are float64 arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost = np.maximum(cost, 1e-9)
    
    # Cost-adjusted LCB:
    # Standard LCB: -mu + beta * sigma
    # Cost-adjusted: -mu + beta * sigma / cost
    # This penalizes high-cost regions more heavily in the exploration term
    
    # Choose beta adaptively based on the scale of the problem
    # For rosenbrock on [-2,2]^2, typical values are in the range [0, ~100]
    # f_best starts high and decreases
    
    # Use a beta that balances exploitation and exploration
    # Scale beta by the current f_best to adapt to the optimization progress
    # When f_best is large (early), explore more; when small (late), exploit more
    
    # Base beta for LCB
    beta_base = 2.0
    
    # Adaptive scaling: as f_best decreases, reduce exploration
    # Normalize f_best to [0, 1] range approximately
    # Rosenbrock minimum is 0, initial values can be up to ~100
    f_scale = max(abs(f_best), 1e-6)
    beta = beta_base * (1.0 + 0.5 * np.log1p(f_scale))
    
    # Compute cost-adjusted LCB score
    # We want higher scores for better candidates
    # LCB lower bound: mu - beta * sigma / cost
    # Acquisition: -LCB = -mu + beta * sigma / cost
    
    sigma_inv_cost = sigma / cost
    score = -mu + beta * sigma_inv_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
