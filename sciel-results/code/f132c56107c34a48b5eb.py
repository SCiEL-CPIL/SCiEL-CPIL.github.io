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

    # Expected Improvement (EI) for minimization with adaptive exploration parameter xi
    # xi scales linearly with posterior uncertainty (sigma)
    # This adapts the exploitation/exploration trade-off dynamically
    
    # Ensure sigma is non-negative and finite to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive exploration parameter
    # Using a small coefficient to slightly bias towards exploitation in uncertain regions
    xi = 0.1 * sigma_safe
    
    # Calculate z-score with adaptive xi
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / sigma_safe
    
    # Compute CDF and PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI with adaptive xi
    # EI(xi) = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
