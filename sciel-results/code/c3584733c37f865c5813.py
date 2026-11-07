# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    variance-based exploration term to balance exploitation and exploration.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) term for exploration
    # We add a small exploration bonus proportional to uncertainty
    beta = 0.1
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB. 
    # EI is already scaled by the magnitude of improvement.
    # We use a weighted sum. Since EI can be large when improvement is large,
    # and UCB is in the same units (objective value), we need to be careful.
    # Actually, EI is in objective units. UCB is in objective units.
    # Let's stick to a robust combination.
    # A common approach is to use EI as the primary score and add a small
    # noise term to break ties or encourage exploration in flat regions.
    # However, pure EI can be greedy.
    # Let's try a hybrid: score = EI + gamma * sigma
    # This encourages exploring high-variance regions slightly.
    gamma = 0.05
    score = ei + gamma * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
