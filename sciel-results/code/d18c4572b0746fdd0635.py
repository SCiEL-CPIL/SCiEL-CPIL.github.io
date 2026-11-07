import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Handle edge cases where sigma is 0
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: Divide by cost
    # Cost is in [1, 2]
    score = ei / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
