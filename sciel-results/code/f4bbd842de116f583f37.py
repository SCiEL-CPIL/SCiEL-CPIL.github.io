import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - kappa * sigma
    # So we maximize -(mu - kappa * sigma) = -mu + kappa * sigma
    
    # Handle small sigma to avoid numerical issues, though LCB is stable
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Exploration parameter
    kappa = 2.0
    
    # Score: higher is better
    score = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
