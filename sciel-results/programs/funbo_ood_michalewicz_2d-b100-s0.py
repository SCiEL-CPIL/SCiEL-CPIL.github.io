"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu

    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize: -mu + kappa * sigma
    # This favors points with low predicted mean (exploitation)
    # and high uncertainty (exploration)
    
    # Determine exploration parameter kappa
    # A fixed kappa is often more robust than adaptive ones in small-sample BO
    # 1.0 is a standard starting point
    kappa = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate UCB score
    # Higher score means more likely to be selected
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
