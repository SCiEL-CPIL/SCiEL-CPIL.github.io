"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # Maximizes: (beta * sigma - mu) / cost
    # beta controls exploration vs exploitation
    beta = 3.0
    sigma = np.maximum(sigma, 1e-8)
    return (beta * sigma - mu) / cost
# EVOLVE-BLOCK-END
