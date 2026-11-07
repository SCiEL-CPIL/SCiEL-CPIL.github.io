"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # Score = -mu + beta * sigma
    # Higher score is better.
    beta = 1.0
    sigma_safe = np.maximum(sigma, 1e-9)
    return -mu + beta * sigma_safe
# EVOLVE-BLOCK-END
