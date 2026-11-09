"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 325.0  # 5 seconds margin before 330s

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > DEADLINE:
        # Return a score that favors the current best (exploitation)
        return -mu
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Calculate the term (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Phi(z) and phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI formula
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
