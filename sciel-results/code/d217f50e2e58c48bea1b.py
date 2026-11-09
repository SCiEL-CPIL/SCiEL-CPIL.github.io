"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with cost-awareness. The blend is tuned to balance exploitation and exploration
    for the multimodal Hartmann 6d landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no negative or zero costs
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB: exploit promising regions while exploring uncertain areas
    # beta = 1.5 balances exploration and exploitation well for Hartmann 6d
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # Blend EI and UCB with 0.6/0.4 ratio
    # EI captures expected improvement, UCB captures uncertainty
    blended = 0.6 * ei + 0.4 * ucb
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # This ensures we get more value per unit cost
    cost_aware = blended / cost
    
    # Add a small exploration bonus for very uncertain regions
    # This helps escape local minima in the multimodal landscape
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-9:
        sigma_mean = 1.0
    
    # Exploration bonus: normalized uncertainty
    exploration_bonus = 0.1 * (sigma / sigma_mean)
    
    # Combine cost-aware score with exploration bonus
    base_score = cost_aware + exploration_bonus
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(base_score), np.maximum(base_score, 0.0), 0.0)
    
    # Ensure all values are finite and non-negative
    base_score = np.where(np.isfinite(base_score), base_score, 0.0)
    base_score = np.maximum(base_score, 0.0)
    
    return base_score
# EVOLVE-BLOCK-END
