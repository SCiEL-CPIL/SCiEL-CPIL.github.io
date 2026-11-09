import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: EI / cost
    # This is the standard cost-aware acquisition function
    result = ei / cost
    
    # Ensure all values are finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
