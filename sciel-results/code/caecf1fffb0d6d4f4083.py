import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimization.
    Higher score means more likely to improve upon f_best.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
