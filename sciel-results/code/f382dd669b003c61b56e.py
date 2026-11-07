"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization.
    # Higher score means more promising (lower mu, higher sigma).
    beta = 2.0
    return mu - beta * sigma
# EVOLVE-BLOCK-END
