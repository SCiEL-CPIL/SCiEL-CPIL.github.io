import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Knowledge Gradient: Maximize expected reduction in f_best / cost^0.5
    # Using square root of cost to reduce the penalty for moderately expensive
    # regions (like the center of the domain) while still preferring cheaper
    # points when gains are similar.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    gain = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware scoring: Divide by square root of cost
    cost_safe = np.maximum(cost, 1e-9)
    score = gain / np.sqrt(cost_safe)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
