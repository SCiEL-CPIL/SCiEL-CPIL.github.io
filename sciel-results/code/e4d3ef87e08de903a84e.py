import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using xi for improvement threshold
    xi = 0.01
    z = (f_best - xi - mu) / sigma
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost penalty
    # Bias towards center of domain (cost 1.5)
    # Quadratic penalty
    alpha = 0.1
    penalty = alpha * (cost - 1.5)**2
    
    scores = ei - penalty
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
