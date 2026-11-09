import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    sigma = np.maximum(sigma, 1e-6)
    
    # Thompson Sampling with cost-awareness
    # Sample from posterior: y ~ N(mu, sigma^2)
    # Score = -y / cost (minimize y, so maximize -y/cost)
    # But we need to balance exploration (high sigma) and exploitation (low mu)
    # Thompson Sampling naturally does this by sampling
    
    # Use a fixed number of Monte Carlo samples for stability
    n_samples = 20
    rng = np.random.RandomState(42 + _call_count)
    
    # Sample from the Gaussian posterior
    samples = rng.normal(mu, sigma, size=(n_samples, len(mu)))
    
    # For each candidate, compute the expected minimum cost-normalized objective
    # cost_normalized_obj = sample / cost
    # We want to minimize this, so we maximize -min(cost_normalized_obj)
    cost_normalized = samples / (cost[None, :] + 1e-9)
    
    # Take the minimum across samples for each candidate
    min_cost_normalized = np.min(cost_normalized, axis=0)
    
    # Score is negative of this (we maximize)
    scores = -min_cost_normalized
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
