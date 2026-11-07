"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # Higher score means more promising
    # Score = -mu + beta * sigma
    # We choose beta = 1.5 as a balance between exploitation and exploration
    beta = 1.5
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
