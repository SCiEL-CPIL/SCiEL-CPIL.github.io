"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost.
    # EI balances exploration and exploitation; dividing by cost
    # favors cheap evaluations in cost-aware BO.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    return ei / cost
# EVOLVE-BLOCK-END
