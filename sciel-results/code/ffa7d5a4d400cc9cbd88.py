import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Knowledge Gradient: Maximize expected reduction in f_best / cost
    # E[reduction] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    gain = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    score = gain / (cost + 1e-9)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
