"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a cost-aware penalty and
    a small uncertainty bonus to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: penalize expensive evaluations
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost-aware EI: divide by a function of cost
    # Use a soft penalty that doesn't completely zero out expensive points
    cost_penalty = 1.0 + cost_norm  # Range [1, 2]
    ei_cost = ei / cost_penalty
    
    # Uncertainty bonus: encourages exploration in uncertain regions
    # Scale sigma to be comparable to EI
    sigma_scale = np.max(sigma) + 1e-9
    unc_bonus = 0.1 * (sigma / sigma_scale)
    
    # Combine: primarily EI, with small uncertainty bonus
    scores = ei_cost + unc_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
