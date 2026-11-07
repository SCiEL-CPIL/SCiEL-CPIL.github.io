import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon for numerical stability
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score
    # Soften the cost penalty by using the square root
    # This reduces the relative penalty for expensive points,
    # allowing exploration of high-cost regions if they have high EI.
    score = ei / np.sqrt(cost + eps)
    
    return score
