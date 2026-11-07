"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    - mu: (2000,) mean predictions from GP
    - sigma: (2000,) standard deviation predictions from GP
    - f_best: scalar, best observed value
    - cost: (2000,) evaluation costs for each candidate
    
    Returns:
    - (2000,) acquisition scores (higher = better)
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term: balance exploitation and exploration
    # Use a beta parameter that adapts based on the scale of uncertainty
    # Higher beta = more exploration
    beta = 2.0
    
    # UCB = mu - beta * sigma (we minimize, so lower mu and higher sigma are good)
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But also add a small exploration bonus for uncertain regions
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combine EI and UCB with cost awareness
    # EI is already in objective units, UCB is in objective units
    # Weight them and divide by cost
    ei_cost_aware = ei / cost_safe
    ucb_cost_aware = ucb / cost_safe
    
    # Blend: 70% exploitation (EI), 30% exploration (UCB)
    score = 0.7 * ei_cost_aware + 0.3 * ucb_cost_aware
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
