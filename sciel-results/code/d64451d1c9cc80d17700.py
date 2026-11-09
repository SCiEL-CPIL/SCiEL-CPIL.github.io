"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score, so we prefer low mu and high sigma.
    # score = -mu + beta * sigma
    beta = 2.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
