import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle zero variance to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized difference
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of standard normal
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite scores
    return ei
