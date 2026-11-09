"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_deadline = time.time() + 280.0  # safety margin before 300s wall-clock limit


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-LCB divided by cost.
    
    Uses a low quantile of the predictive distribution (10th percentile)
    to balance exploitation and exploration, then normalizes by cost
    to prefer cheaper evaluations.
    """
    # Check deadline
    if time.time() > _deadline:
        return np.zeros_like(mu)
    
    # Protect against division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Quantile-LCB: use 10th percentile of the predictive distribution
    # This is equivalent to: mu - z * sigma where z is the z-score for 10%
    # z for 10th percentile is approximately 1.2816
    z_quantile = 1.2816
    
    # Lower confidence bound (lower is better for minimization)
    lcb = mu - z_quantile * sigma_safe
    
    # We want higher scores to be evaluated, so we negate the LCB
    # (since lower LCB means more promising for minimization)
    score = -lcb
    
    # Normalize by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    
    result = score / cost_safe
    
    # Ensure all values are finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
