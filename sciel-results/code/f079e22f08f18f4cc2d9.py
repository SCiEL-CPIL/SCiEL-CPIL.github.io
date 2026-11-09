"""Cost-aware acquisition function using Expected Improvement normalized by cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 325.0  # Safety margin of 5s before the 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by the evaluation cost.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    # Deadline guard: If we are close to the limit, return a fast, conservative score
    # to ensure we finish safely. -mu/cost prioritizes exploitation of the current
    # best region while respecting cost.
    if time.time() > _DEADLINE - 2.0:
        cost_safe = np.where(cost > 1e-9, cost, 1.0)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    
    # Avoid division by zero in sigma for z calculation
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)

    # Compute Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf and norm.pdf are stable, but clipping prevents inf/nan in rare edge cases
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Cost-awareness: Divide by cost
    # This prioritizes candidates that offer high expected improvement per unit cost.
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
