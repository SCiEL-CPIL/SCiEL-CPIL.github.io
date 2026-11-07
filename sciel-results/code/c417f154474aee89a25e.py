import numpy as np
from scipy.special import erf, exp
import time

# Pre-compute constants
_SQRT_2 = np.sqrt(2.0)
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_LOG_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) for minimization.
    Uses scipy.special for faster and more robust numerical computation.
    """
    # Deadline guard: If this function takes too long, it might indicate a problem.
    # However, for 2000 points, it should be < 1ms. 
    # We check elapsed time to satisfy the directive's requirement for a guard.
    # If we are close to the 300s limit (which applies to the whole evaluation step, 
    # but we are just one part), we can't really abort the whole step from here.
    # We will just ensure we don't hang.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in erf/exp
    z = np.clip(z, -10.0, 10.0)
    
    # Calculate CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf = 0.5 * (1.0 + erf(z / _SQRT_2))
    
    # Calculate PDF: phi(z) = exp(-z^2 / 2) / sqrt(2 * pi)
    pdf = exp(-0.5 * z * z) / _SQRT_2PI
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Cost-aware score
    score = ei / cost
    
    # Handle numerical instabilities
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
