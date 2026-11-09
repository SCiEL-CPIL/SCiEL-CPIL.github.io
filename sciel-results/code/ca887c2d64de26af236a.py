"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with small exploration term.
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Add a small term to encourage exploration: -lambda * sigma
    # This helps escape local minima in the Beale function.
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus proportional to uncertainty
    # This encourages evaluating points with high uncertainty
    exploration_weight = 0.1
    score = ei - exploration_weight * sigma
    
    return score
# EVOLVE-BLOCK-END
