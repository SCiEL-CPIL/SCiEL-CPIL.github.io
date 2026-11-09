import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Calculate the z-score for the Probability of Improvement
    # PI = P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Compute Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher PI and lower cost is better
    # cost is given as (2000,), values in [1, 2]
    scores = pi / cost
    
    # Ensure outputs are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
