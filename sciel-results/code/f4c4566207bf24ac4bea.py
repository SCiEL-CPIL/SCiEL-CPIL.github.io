"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _safe_divide(a, b, eps=1e-9):
    """Safe element-wise division with epsilon guard."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement divided by the square root of the cost.
    This reduces the bias towards low-cost regions (corners) compared to
    dividing by cost directly, allowing better exploration of the center
    where the global minimum of Griewank is located.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = _safe_divide(diff, sigma)
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for very large values
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by square root of cost to reduce bias towards cheap corners
    # Cost is in [1, 2]. sqrt(Cost) is in [1, 1.414].
    # This makes the acquisition function less sensitive to cost differences,
    # allowing the GP's predictive uncertainty and mean to drive the search
    # towards the true minimum more effectively.
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = np.sqrt(cost_safe)
    
    score = _safe_divide(ei, cost_factor, eps=1e-9)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
