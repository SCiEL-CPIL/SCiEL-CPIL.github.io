import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so we maximize -LCB / cost
    # -LCB = beta * sigma - mu
    
    # Safety clamp for sigma and cost
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Exploration parameter
    beta = 1.0
    
    # Calculate LCB score (higher is better)
    # Numerator: beta * sigma - mu
    # This is high when mu is low (good prediction) or sigma is high (uncertainty)
    numerator = beta * sigma - mu
    
    # Cost-aware score
    score = numerator / cost
    
    return score
