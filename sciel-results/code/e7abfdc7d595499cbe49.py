"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track wall-clock time to avoid timeout
_start_time = time.time()
_DEADLINE = 330.0
_TIME_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Uses LCB to identify points where the lower bound on the function value
    is below the current best, indicating potential for improvement.
    Applies a quadratic cost penalty to strongly prioritize cheap evaluations,
    allowing more samples within the fixed budget of 30.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: return a fast fallback if we're near the time limit
    elapsed = time.time() - _start_time
    if elapsed > (_DEADLINE - _TIME_MARGIN):
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-8)
        return 1.0 / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost_arr, 1e-8)
    
    # Compute LCB for minimization: LCB(x) = mu(x) - kappa * sigma(x)
    # We want to find points where LCB is well below f_best
    kappa = 1.0
    lcb = mu - kappa * sigma_safe
    
    # Potential improvement: how much lower the LCB is than f_best
    # This is positive when there's potential improvement
    improvement = f_best - lcb
    
    # Clip to non-negative: only reward points with actual potential
    improvement = np.maximum(improvement, 0.0)
    
    # Quadratic cost penalty: strongly prefer cheap evaluations
    # This allows more evaluations within the cost budget of 30
    score = improvement / (cost_safe ** 2)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
