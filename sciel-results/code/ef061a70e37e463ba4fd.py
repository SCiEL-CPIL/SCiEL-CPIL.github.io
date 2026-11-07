"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Minimum-variance acquisition: score candidates by posterior standard deviation only.
    # This forces exploration of unexplored regions by prioritizing high uncertainty,
    # ignoring the mean to prevent premature convergence to local noise minima.
    # Ensure sigma is finite and non-negative for valid scoring.
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    return sigma_safe
# EVOLVE-BLOCK-END
