import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: simple LCB
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        return -mu - 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: If f_best - mu is large positive, EI is large.
    # If f_best - mu is negative (mu < f_best), EI is still positive due to the sigma term?
    # Let's check the formula.
    # Standard EI for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # Let Z ~ N(mu, sigma^2).
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    # Handle large z to avoid overflow in exp
    # Phi(z) is 1 for large z, 0 for large negative z
    # phi(z) is 0 for large |z|
    
    # Use scipy's norm
    # cdf = norm.cdf(z)
    # pdf = norm.pdf(z)
    
    # To be safe against numerical issues, we can use erfc for cdf if needed, but scipy is fine.
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
