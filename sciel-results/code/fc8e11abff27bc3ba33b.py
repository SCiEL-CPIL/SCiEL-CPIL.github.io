import time
import numpy as np
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback to greedy exploitation if time is critical
        # Return -mu so that the lowest mu (best value) gets the highest score
        return -mu 

    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    # For |z| > 10, Phi(z) is essentially 0 or 1, and phi(z) is negligible
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate CDF and PDF
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    # Compute EI
    # Term 1: (f_best - mu) * Phi(z)
    # Term 2: sigma * phi(z)
    ei_scores = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure scores are finite
    # If mu or sigma were inf/nan, handle them
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
