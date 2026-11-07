"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + gamma * sigma
    # Higher score is better.
    # gamma controls the exploration-exploitation trade-off.
    gamma = 2.0
    score = -mu + gamma * sigma
    return score
# EVOLVE-BLOCK-END
