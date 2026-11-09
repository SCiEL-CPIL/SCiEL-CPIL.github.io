"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize the score to select the next point.
    # Score = -Mean + Beta * StdDev
    # Higher score means more promising (lower expected value or higher uncertainty).
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
