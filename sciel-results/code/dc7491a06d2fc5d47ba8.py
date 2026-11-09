import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) approach
    # We want to minimize f(x).
    # LCB = mu - beta * sigma
    # We want to find points with low LCB.
    # Acquisition score to maximize: -LCB / cost = (beta * sigma - mu) / cost
    
    beta = 2.0
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate LCB score
    # If mu is very high, -mu is very negative.
    # If sigma is high, beta*sigma is positive.
    # We prefer low mu and high sigma (exploration).
    lcb_score = beta * sigma_safe - mu
    
    # Normalize by cost
    score = lcb_score / cost_safe
    
    return score
