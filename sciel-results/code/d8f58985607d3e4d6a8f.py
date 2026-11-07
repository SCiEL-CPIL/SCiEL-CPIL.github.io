"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Minimum-variance acquisition: score candidates by posterior standard deviation.
    # Higher scores correspond to higher uncertainty, forcing exploration of unexplored regions.
    
    # Ensure sigma is finite and non-negative
    scores = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
