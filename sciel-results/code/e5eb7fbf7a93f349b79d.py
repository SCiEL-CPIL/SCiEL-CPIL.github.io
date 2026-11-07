# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    EI(x) = E[max(f_best - f(x), 0)]
    For f(x) ~ N(mu, sigma^2), 
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    Score = EI(x) / cost(x)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Higher EI per unit cost is better
    # cost is always >= 1 in this problem (1 + mean(x)), so no division by zero risk
    score = ei / np.maximum(cost, 1e-10)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
