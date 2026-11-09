import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    Uses Expected Improvement (EI).
    """
    global _start_time

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return simple PMM to avoid timeout
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)

    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    z = (f_best - mu) / sigma
    
    # Use scipy.stats.norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Numerical safety: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)

    return ei
