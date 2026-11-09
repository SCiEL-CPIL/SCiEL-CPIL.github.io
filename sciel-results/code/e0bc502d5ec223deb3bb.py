import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (POI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF
    poi = norm.cdf(z)
    
    # Cost-aware score: higher POI per unit cost is better
    score = poi / cost_safe
    
    return score
