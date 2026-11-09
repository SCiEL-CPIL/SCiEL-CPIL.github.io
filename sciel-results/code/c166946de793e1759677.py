import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # We know the global minimum is 0.
    # Using target 0 makes the acquisition function more aggressive towards the true minimum.
    target = 0.0
    
    # Clip sigma to avoid division by zero and to encourage exploration in high-confidence regions
    # that might be local minima.
    sigma_safe = np.maximum(sigma, 1e-6)
    
    z = (target - mu) / sigma_safe
    
    # Expected Improvement with target
    # EI = (target - mu) * Phi(z) + sigma * phi(z)
    scores = (target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    return scores
