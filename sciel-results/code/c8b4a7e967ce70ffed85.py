"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_TIMEOUT_MARGIN = 25.0  # return at least 25s before deadline


def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Returns 2000 finite scores (higher = evaluate next).
    
    Strategy: Expected Improvement per unit cost, with robust handling
    of numerical issues and a time guard.
    """
    if _check_timeout():
        # Return a simple fallback: prefer low cost and low mu
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: non-positive sigma
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (can be slightly negative due to numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
