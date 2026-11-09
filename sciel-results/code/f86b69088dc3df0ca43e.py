"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB).
    
    UCB balances exploitation (low mean) and exploration (high uncertainty).
    For minimization, we maximize the score: -mu + beta * sigma.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Avoid division by zero or negative sigma issues (though sigma should be positive)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter. 
    # A value of 1.0 is a standard starting point for balanced exploration/exploitation.
    # We can adjust this if needed.
    beta = 1.0
    
    # Score = -mu + beta * sigma
    # Higher score indicates a better candidate to evaluate next.
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
