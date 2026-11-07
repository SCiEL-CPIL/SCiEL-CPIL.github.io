import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and handle zeros to avoid division by zero in z-score
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is given as an array of shape (2000,)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
