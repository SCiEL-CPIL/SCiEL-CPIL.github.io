import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Compute z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware penalty: softer than division by cost
    # cost is in [1, 2]. cost - 1 is in [0, 1].
    # We want to penalize higher cost, but not too much.
    # Let's use a multiplicative factor.
    # If cost=1, factor=1. If cost=2, factor=1/(1+0.1) = 0.909.
    # This is a mild penalty.
    cost_penalty = 1.0 / (1.0 + 0.1 * (cost - 1.0))
    
    scores = ei * cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
