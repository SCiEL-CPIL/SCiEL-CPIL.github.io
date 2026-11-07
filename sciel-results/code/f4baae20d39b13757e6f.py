"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # Score = -mu + beta * sigma.
    # Higher score indicates a more promising candidate (lower expected value or high uncertainty).
    beta = 3.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
