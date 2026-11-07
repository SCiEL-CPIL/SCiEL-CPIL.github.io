"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean of GP predictions.
    sigma : ndarray of shape (2000,)
        Standard deviation of GP predictions.
    f_best : float
        Best observed value so far.
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adjustment: divide by cost to get EI per unit cost
    # Use a soft cost penalty to avoid extreme values
    # cost is in [1, 2] since x scaled to [0,1] gives mean in [0,1]
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    
    # Cost-aware EI: normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    cost_aware_ei = ei / cost_safe
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps balance exploitation and exploration
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
