"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, keep 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback to avoid timeout
        return np.ones(2000) / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Also add a small exploration bonus based on uncertainty
    # This helps balance exploitation and exploration
    exploration_bonus = 0.1 * sigma / (sigma + 1.0)
    
    scores = ei / cost + exploration_bonus / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
