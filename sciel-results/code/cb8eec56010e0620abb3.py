"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting high-uncertainty regions
    # This helps escape local minima in multimodal functions like Hartmann 3D
    exploration_term = 0.2 * sigma
    
    return ei + exploration_term
# EVOLVE-BLOCK-END
