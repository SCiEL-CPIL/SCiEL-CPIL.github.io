"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound per unit cost
    # Higher score is better. We want to minimize f, so we maximize -mu + beta*sigma.
    # Cost is in [1, 2].
    beta = 2.0
    ucb = -mu + beta * sigma
    return ucb / cost
# EVOLVE-BLOCK-END
