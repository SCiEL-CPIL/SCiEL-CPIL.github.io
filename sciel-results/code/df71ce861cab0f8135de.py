import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_PROGRAM_START = time.time()
_DEADLINE = 300.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are close to the limit, switch to pure exploitation
    # to ensure we finish within the time limit.
    if time.time() - _PROGRAM_START > _DEADLINE - 0.5:
        return -mu

    # Ensure sigma is non-negative and avoid division by zero
    # A small epsilon prevents numerical issues in the Z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z score: (f_best - mu) / sigma
    # If sigma is very small, Z will be large if mu < f_best (exploit) or small if mu > f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Vectorized computation
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
