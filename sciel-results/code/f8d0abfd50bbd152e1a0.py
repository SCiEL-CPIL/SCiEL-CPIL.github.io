"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and a cost penalty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Convert inputs to numpy arrays for consistent handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive (avoid division by zero)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scaling: divide by cost to get improvement per unit cost
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty to balance exploitation/exploration
    # This helps when EI is near zero but uncertainty is high
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-10:
        exploration_bonus = 0.1 * sigma_safe / max_sigma
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    # Combine EI per cost with exploration bonus
    scores = ei_per_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
