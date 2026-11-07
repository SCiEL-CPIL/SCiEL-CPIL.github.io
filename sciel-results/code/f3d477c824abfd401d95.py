# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Parameters:
    mu: array of posterior means, shape (N,)
    sigma: array of posterior standard deviations, shape (N,)
    f_best: scalar, the best observed value so far
    
    Returns:
    scores: array of EI scores, shape (N,), higher is better
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks: ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate z-score
    # For minimization, improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10) # Clip to avoid numerical issues with extreme values
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * P(f(x) < f_best) + E[f_best - f(x) | f(x) >= f_best] * P(f(x) >= f_best)
    # Simplified formula: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Time guard (unlikely to be needed for vectorized ops, but good practice)
    if time.time() - t0 > 250:
        # Fallback to a simple uncertainty-based score if calculation is stuck (shouldn't happen)
        ei = sigma
        
    return ei
# EVOLVE-BLOCK-END
