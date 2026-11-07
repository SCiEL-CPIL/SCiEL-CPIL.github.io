"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # Beta is a trade-off parameter.
    # For low-budget optimization, a higher beta encourages exploration.
    # A value of 2.0 is a common starting point.
    beta = 2.0
    return mu + beta * sigma
# EVOLVE-BLOCK-END
