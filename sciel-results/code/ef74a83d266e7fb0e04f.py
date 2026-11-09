"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # Maximizes -mu + beta * sigma.
    # beta controls the exploration/exploitation trade-off.
    beta = 1.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
