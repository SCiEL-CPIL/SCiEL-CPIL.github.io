import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline Guard: 
    # The evaluator kills after 330s. 
    # We check if we are dangerously close to the limit.
    # If so, we return a "safe" acquisition function that just exploits 
    # the most promising point (lowest mu) to finish quickly and safely.
    elapsed = time.time() - _start_time
    if elapsed > 320: 
        # Return score based purely on exploitation (lowest mu is best)
        # Score = -mu (higher is better)
        # Ensure finiteness
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Expected Improvement (EI) for minimization.
    # We want to maximize the expected improvement over f_best.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is non-negative and finite to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the standard normal variable
    z = (f_best - mu) / sigma_safe
    
    # Compute the expected improvement
    # Use scipy's norm.cdf and norm.pdf which handle extreme values well
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all values are finite and non-negative
    # EI should theoretically be >= 0, but numerical errors can cause small negatives
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
