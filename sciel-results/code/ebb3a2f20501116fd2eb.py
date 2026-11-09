import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > 300:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to avoid division by zero and handle non-finite values
    sigma = np.where(sigma > 1e-10, sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # Let s = f_best - mu(x)
    # If s <= 0, EI = 0
    # If s > 0, EI = s * Phi(s/sigma) + sigma * phi(s/sigma)
    
    s = f_best - mu
    z = s / sigma
    
    # Initialize scores to 0
    scores = np.zeros_like(mu)
    
    # Only compute for s > 0
    mask = s > 0
    
    if np.any(mask):
        z_masked = z[mask]
        s_masked = s[mask]
        sigma_masked = sigma[mask]
        
        # Calculate PDF and CDF of standard normal
        # Using scipy.stats.norm
        phi_z = norm.pdf(z_masked)
        cdf_z = norm.cdf(z_masked)
        
        # Compute EI
        ei_vals = s_masked * cdf_z + sigma_masked * phi_z
        
        scores[mask] = ei_vals
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
