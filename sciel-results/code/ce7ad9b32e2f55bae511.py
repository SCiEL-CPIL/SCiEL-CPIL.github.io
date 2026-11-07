import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the "potential" of being lower than f_best.
    # Score proportional to (beta * sigma - mu) / cost
    beta = 2.0
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    lcb_term = beta * sigma - mu
    # Avoid division by zero in cost, though cost >= 1
    safe_cost = np.maximum(cost, 1e-8)
    
    scores = lcb_term / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
