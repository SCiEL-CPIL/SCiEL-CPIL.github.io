"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score to find the next point.
    # For minimization, low mu is good, high sigma is good (exploration).
    # Score = -mu + beta * sigma
    # We choose beta = 2.5 to balance exploration and exploitation.
    beta = 2.5
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
