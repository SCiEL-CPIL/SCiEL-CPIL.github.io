# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses Upper Confidence Bound (UCB) acquisition function.
    UCB balances exploration and exploitation by adding an uncertainty bonus
    to the negative mean (since we are minimizing).
    Score = -mu + beta * sigma
    Higher scores are preferred.
    """
    # Avoid division by zero or numerical issues, though UCB doesn't divide by sigma directly
    # But we should handle edge cases where sigma is 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration parameter
    # A fixed beta is simple. 
    # In minimization, we want low mu. So -mu is the exploitation term.
    # We add beta * sigma for exploration.
    beta = 1.0
    
    # Calculate UCB score
    # If sigma is very small, the term is small.
    # If sigma is large, the term is large.
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # If all scores are -inf (shouldn't happen with safe sigma), replace with 0
    if np.all(score == -np.inf):
        score = np.zeros_like(score)
        
    return score
# EVOLVE-BLOCK-END
