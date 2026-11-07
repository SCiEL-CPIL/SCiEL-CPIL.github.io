import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add small exploration bonus to encourage diversity
    # This helps if the GP is slightly overconfident
    exploration_bonus = 0.1 * sigma_safe
    
    score = (ei + exploration_bonus) / cost
    
    return score
