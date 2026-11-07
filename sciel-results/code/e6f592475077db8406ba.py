"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) acquisition function.
    # Favors points with a high likelihood of surpassing the current best.
    # Add a small jitter to sigma to avoid division by zero.
    sigma_safe = sigma + 1e-9
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)
# EVOLVE-BLOCK-END
