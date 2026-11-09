import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic lambda
    # f_best is the lowest observed value.
    # As we approach 0, lambda increases.
    # If f_best is negative (shouldn't happen for Rosenbrock, but safe), use abs.
    # Rosenbrock min is 0.
    lambda_val = 1.0 / (1.0 + abs(f_best))
    
    # Exponential penalty
    penalty = np.exp(-lambda_val * cost)
    
    scores = ei * penalty
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
