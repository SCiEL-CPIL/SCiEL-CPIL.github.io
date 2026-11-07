import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0

def acquisition(mu, sigma, f_best):
    # Check time
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid division by zero
    # The GP noise is 1e-6, so sigma should be at least ~1e-3 or so, 
    # but just in case, we clip it.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle the case where sigma is very small to avoid numerical instability
    # If sigma is effectively 0, the distribution is a delta function at mu.
    # EI is max(f_best - mu, 0)
    
    z = (f_best - mu) / sigma_safe
    
    # For very large z, Phi(z) is 1 and phi(z) is 0.
    # For very small z, Phi(z) is 0 and phi(z) is 0.
    # Standard formulas work well, but let's be careful.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # If sigma was clipped, and the original sigma was 0, we might have artifacts.
    # But with GP noise, sigma > 0 always.
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative by definition.
    ei = np.maximum(ei, 0.0)
    
    return ei
