import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize the function, so we look for the lowest possible value
    # LCB = mu - kappa * sigma
    # We maximize the score, which corresponds to minimizing LCB / cost
    # Score = -LCB / cost = (kappa * sigma - mu) / cost
    
    # Choose a kappa for exploration. 1.0 is a standard starting point.
    # A slightly higher value might be better for global exploration.
    kappa = 1.5
    
    ucb_score = kappa * sigma_safe - mu
    
    # Cost-aware score
    scores = ucb_score / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
