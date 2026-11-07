"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost (modified: sqrt cost)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use sqrt(cost) to reduce penalty for expensive points (near minimum)
    # This structural change favors the center (high cost, low f) more than linear cost division
    scores = ei / np.sqrt(cost)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
