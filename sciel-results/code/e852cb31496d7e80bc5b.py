import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) divided by an aggressive cost penalty.
    """
    # Ensure sigma is non-negative and not zero to avoid division errors
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Apply aggressive cost penalty
    # cost is in [1, 2]. (cost - 0.5) is in [0.5, 1.5].
    # Dividing by this amplifies the preference for low-cost regions.
    cost_penalty = cost - 0.5
    cost_penalty = np.maximum(cost_penalty, 1e-9) # Safety
    
    score = ei / cost_penalty
    
    return score
