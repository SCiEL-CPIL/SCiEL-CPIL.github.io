"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with a small exploration bonus
    # z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage diversity
    # This helps escape local optima in the early stages
    # Use a small coefficient to not dominate the EI signal
    exploration_bonus = 0.01 * sigma_safe
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
