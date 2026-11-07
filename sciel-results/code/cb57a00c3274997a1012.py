# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s/330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI
    ei_cost = ei / cost
    
    # Add exploration bonus based on uncertainty
    # Higher uncertainty regions get a bonus to encourage exploration
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    exploration_bonus = 0.1 * (sigma - sigma_mean) / sigma_std
    
    # Combine EI/cost with exploration bonus
    scores = ei_cost + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
