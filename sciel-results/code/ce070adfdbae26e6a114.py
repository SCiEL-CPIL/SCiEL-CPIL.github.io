import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
