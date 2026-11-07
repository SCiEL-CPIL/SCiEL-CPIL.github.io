"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f, so we maximize -mu + beta * sigma.
    # Higher score means we evaluate next.
    beta = 2.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
