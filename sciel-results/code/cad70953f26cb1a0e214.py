import numpy as np
from scipy.stats import norm
import time

# Global variable to track the start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: The evaluator kills the program after 330s.
    # We use a safety margin of 10s. If we are too close to the limit,
    # we return a simple exploitation score to ensure we don't hang or timeout
    # in the middle of a complex calculation (though EI is fast, this satisfies the directive).
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to simple exploitation (lowest mean)
        scores = -mu
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Expected Improvement (EI) Acquisition Function
    # For minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate standard normal CDF (Phi) and PDF (phi)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Calculate Expected Improvement
    # Note: (f_best - mu) is the expected improvement magnitude if we ignore uncertainty
    # The term sigma * phi(z) accounts for the uncertainty
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ei
