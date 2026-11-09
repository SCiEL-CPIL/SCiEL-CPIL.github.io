import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)

    # Thompson Sampling Heuristic
    # Sample a function value from the GP posterior for each candidate
    # f_sample = mu + sigma * z, where z ~ N(0, 1)
    # We want to minimize f, so we want low f_sample.
    # The acquisition score should be higher for better candidates.
    # Therefore, score = -f_sample / cost
    
    z = np.random.randn(len(mu))
    sampled_values = mu + sigma * z
    
    # Cost-aware score: higher is better
    scores = -sampled_values / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
