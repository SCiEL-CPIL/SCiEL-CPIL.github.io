import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # PI(x) = Phi((f_best - mu(x)) / sigma(x))
    
    # Handle zero or near-zero sigma to avoid numerical instability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the standardized improvement z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PI is the CDF of the standard normal distribution evaluated at z
    pi = norm.cdf(z)
    
    # Ensure all scores are finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
