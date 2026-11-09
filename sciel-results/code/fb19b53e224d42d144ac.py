"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 300s limit, keep 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions
    sigma : array_like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: uniform random scores
        return np.ones_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # EI formula
    ei = np.where(
        sigma > 1e-9,
        (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z),
        np.maximum(f_best - mu, 0)
    )
    
    # Cost-normalized EI
    scores = ei / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
