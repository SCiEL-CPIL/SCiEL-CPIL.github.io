"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    This is a structural change from Expected Improvement (EI).
    UCB balances exploitation (low mu) and exploration (high sigma).
    For minimization, we want to select points with low predicted values
    and high uncertainty, as high uncertainty implies a higher chance
    of finding a value significantly lower than the current best.
    
    Score = beta * sigma - mu
    """
    # Hyperparameter for exploration
    beta = 2.0
    
    # Handle edge cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute UCB score for minimization
    # We want to maximize this score
    # Low mu is good, high sigma is good
    ucb = beta * sigma_safe - mu
    
    # Ensure finite scores
    ucb = np.nan_to_num(ucb, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    
    # We return the scores. The evaluator picks the highest score.
    # If ucb is -inf, it will not be picked.
    return ucb
# EVOLVE-BLOCK-END
