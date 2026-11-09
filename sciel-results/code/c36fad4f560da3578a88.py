import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB with beta = 0.5
    scores = -mu + 0.5 * sigma_safe
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
