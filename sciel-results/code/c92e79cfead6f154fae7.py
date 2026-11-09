import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid: EI + uncertainty bonus for balanced exploration/exploitation
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add uncertainty bonus to encourage exploration in high-variance regions
    # This helps escape local optima in the Beale function landscape
    beta = 0.5
    return ei + beta * sigma_safe
