"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300  # Safety margin of 300s as per prompt

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition score for each candidate point.
    Uses Expected Improvement (EI) with a small exploration bonus.
    
    Args:
        mu: Array of shape (N,) with posterior means.
        sigma: Array of shape (N,) with posterior standard deviations.
        f_best: Scalar, the lowest observed value so far.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return safe zeros if we are out of time
        return np.zeros_like(mu)
        
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploration in high-uncertainty regions
    # This helps break ties and prevents stagnation in flat regions
    exploration_bonus = 0.05 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
