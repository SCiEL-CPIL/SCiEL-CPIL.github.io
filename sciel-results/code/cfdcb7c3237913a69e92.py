import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma close to 0 to avoid division by zero
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    z = (f_best - mu) / sigma_safe
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    return ei
