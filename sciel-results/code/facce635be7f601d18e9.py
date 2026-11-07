"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score, which corresponds to exploring areas
    # with low predicted mean (low mu) and high uncertainty (high sigma).
    # Score = -mu + beta * sigma
    # beta controls the exploration-exploitation trade-off.
    beta = 1.5
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
