"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses Expected Improvement (EI) which is robust and standard.
    """
    start_time = time.time()
    deadline = 330.0
    safety_margin = 5.0

    n = len(mu)
    
    # Handle sigma ~ 0 cases to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Check time
    if time.time() > deadline - safety_margin:
        return ei
    
    return ei

# EVOLVE-BLOCK-END
