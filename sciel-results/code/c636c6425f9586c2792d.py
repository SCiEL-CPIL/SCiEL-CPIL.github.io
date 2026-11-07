import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Thompson Sampling with cost awareness
    # Sample from the posterior: y = mu + sigma * z
    # We want to minimize y, but penalize by cost
    # Score = -y / cost (higher is better)
    # But we need to be careful about division by zero and negative costs
    
    # Generate random sample
    z = np.random.standard_normal(len(mu))
    y_sample = mu + sigma * z
    
    # Cost-aware: divide by cost (lower cost is better)
    # Use a small epsilon to avoid division by zero
    epsilon = 1e-8
    cost_safe = np.maximum(cost, epsilon)
    
    # We want to minimize y_sample, so we negate it
    # And we want to minimize cost, so we divide by cost
    # Score = -y_sample / cost_safe
    # Higher score means we prefer this candidate
    
    scores = -y_sample / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
