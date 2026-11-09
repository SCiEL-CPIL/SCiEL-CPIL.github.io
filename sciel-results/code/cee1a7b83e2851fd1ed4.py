"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Combines Expected Improvement (EI) with a cost penalty.
    
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
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: divide by cost to prefer cheaper evaluations
    # Use a stronger cost penalty to focus on cheaper regions
    # cost is in [1, 2] range (1 + mean(x_scaled_to_01))
    # Using power 1.5 emphasizes cheaper points more
    cost_penalty = np.power(cost, 1.5)
    
    # Score = EI / cost_penalty
    scores = ei / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
