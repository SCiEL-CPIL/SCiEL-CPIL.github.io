import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deterministic cost-penalized Expected Improvement
    # For minimization, EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Compute z-score with small epsilon to avoid division by zero
    sigma_safe = sigma + 1e-9
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost penalty: subtract lambda * cost to guide search toward cheaper regions
    # Cost is in [1, 2]. We use a small lambda to avoid overly aggressive cost bias.
    lambda_cost = 0.1
    cost_penalty = lambda_cost * cost
    
    scores = ei - cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
