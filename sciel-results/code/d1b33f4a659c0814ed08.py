import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: minimize mu - beta * sigma
    # Acquisition score (higher is better): -mu + beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
