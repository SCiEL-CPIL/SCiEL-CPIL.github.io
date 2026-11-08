import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 290.0 # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        # Return a constant score if we are running out of time, 
        # though the evaluator might kill us anyway. 
        # Better to return quickly.
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.nan_to_num(mu, nan=np.median(mu), posinf=np.max(mu), neginf=np.min(mu))
    sigma = np.nan_to_num(sigma, nan=np.median(sigma), posinf=np.max(sigma), neginf=np.min(sigma))
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Clip sigma to avoid numerical issues in division if we were using EI, 
    # but here we just multiply. Still, keep it reasonable.
    sigma = np.clip(sigma, 1e-6, None)
    
    # Cost-aware UCB
    # We want to minimize f, so high score = low expected f + high uncertainty - high cost
    # Score = -mu + beta * sigma - lambda_cost * (cost - 1)
    
    beta = 2.5  # Exploration coefficient
    lambda_cost = 0.5 # Cost penalty coefficient
    
    # Normalize cost penalty to be relative to the range [1, 2]
    # cost - 1 is in [0, 1]
    cost_penalty = lambda_cost * (cost - 1.0)
    
    scores = -mu + beta * sigma - cost_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.max(scores), neginf=np.min(scores))
    
    return scores
