import numpy as np

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to prevent division by zero and numerical instability
    min_sigma = 1e-6
    sigma_safe = np.where(sigma > min_sigma, sigma, min_sigma)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize the value, so we look for low mu and high sigma (uncertainty)
    # The acquisition function to maximize is: -mu + beta * sigma
    # beta controls the exploration/exploitation trade-off
    # A value of 2.0 is often effective for global optimization in early/mid stages
    beta = 2.0
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
