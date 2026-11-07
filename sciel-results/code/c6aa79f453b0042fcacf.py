"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Local Intensive Expected Improvement (LIEI)
    # Clip improvement to non-negative to focus on high-probability improvement zones
    improvement = np.maximum(f_best - mu, 0.0)
    z = (f_best - mu) / (sigma + 1e-9)
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)
# EVOLVE-BLOCK-END
