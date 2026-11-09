"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each point
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e6)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring: EI per unit cost
    # Use a power of cost to more strongly penalize expensive points
    # cost ranges from ~1.0 to ~2.0, so cost^1.5 amplifies the difference
    cost_adj = np.power(cost, 1.5)
    
    scores = ei / cost_adj
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
