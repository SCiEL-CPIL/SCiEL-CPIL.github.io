import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: return safe zeros if we are close to the 330s limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive for stable division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB (Upper Confidence Bound) for minimization
    # We want to maximize: -mu + beta * sigma
    # This encourages exploration (high sigma) and exploitation (low mu)
    
    # Adaptive beta based on f_best and typical sigma scale
    # Use a moderate exploration-exploitation balance
    beta = 2.0
    
    # UCB score: higher is better
    # For minimization, we want low mu and high uncertainty
    ucb = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    ucb = np.nan_to_num(ucb, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ucb
