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

    # Adaptive Expected Improvement with exploration parameter xi
    # xi scales with the mean posterior uncertainty to balance exploration/exploitation
    mean_sigma = np.mean(sigma)
    xi = 0.1 * mean_sigma
    
    # Ensure sigma is non-negative and finite to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI with xi
    # EI_xi(x) = (f_best - xi - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - xi - mu(x)) / sigma(x)
    z = (f_best - xi - mu) / sigma_safe
    
    # Compute CDF and PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI with adaptive xi
    scores = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
