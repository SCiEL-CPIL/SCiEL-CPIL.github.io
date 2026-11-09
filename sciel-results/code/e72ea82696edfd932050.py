import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Soft cost penalty
    # Instead of dividing by cost, we divide by (1 + alpha * cost)
    # This reduces the sensitivity to cost differences
    alpha = 0.1
    scores = ei / (1.0 + alpha * cost)
    
    return scores
