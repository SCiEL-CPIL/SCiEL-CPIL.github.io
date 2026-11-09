"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t_start = time.time()
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI: divide by cost to get improvement per unit cost
    scores = ei / (cost + 1e-9)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps avoid getting stuck in local optima
    # Use a soft exploration term that decays as we find better solutions
    exploration_weight = 0.1
    exploration = exploration_weight * sigma / (cost + 1e-9)
    
    scores = scores + exploration
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
