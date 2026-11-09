import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is not zero to avoid division by zero
    # Use a slightly larger floor to be robust
    sigma_safe = np.maximum(sigma, 1e-6)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values where cdf might lose precision
    # or behave unexpectedly in some lib versions
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z_clipped)
    
    # Ensure finite and within [0, 1] range for stability
    pi = np.clip(pi, 0.0, 1.0)
    
    return pi
