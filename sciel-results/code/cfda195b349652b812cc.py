"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with softer cost penalty.
    # Cost ranges from 1 to 2; using sqrt(cost) reduces over-penalization
    # of expensive regions, allowing broader exploration of the multimodal
    # Shekel 4d landscape.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / np.sqrt(cost)
# EVOLVE-BLOCK-END
