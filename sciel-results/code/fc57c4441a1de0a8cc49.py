import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _DEADLINE
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.ones(len(mu)) / np.maximum(cost, 1e-9)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (normalized)
    pi = norm.cdf(z)
    
    # Exploration bonus: encourages sampling where uncertainty is high
    # Scale sigma relative to the range of mu to make it meaningful
    mu_range = np.ptp(mu) + 1e-8
    sigma_normalized = sigma / mu_range
    
    # Blend: exploitation (EI) + exploration (sigma) 
    # Weight exploration more when EI is small (early in optimization)
    alpha = 0.3  # exploration weight
    
    score = ei + alpha * sigma_normalized * np.ptp(mu)
    
    # Cost-aware: divide by cost
    score = score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 1e-12)
    
    return score
