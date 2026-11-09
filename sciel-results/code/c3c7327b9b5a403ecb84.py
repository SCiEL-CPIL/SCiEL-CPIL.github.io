"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    # Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exponential cost penalty: suppresses expensive regions more strongly
    # cost is in [1, 2] range, exp(-cost) gives stronger differentiation
    cost_penalty = np.exp(-cost)
    
    scores = ei * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # We already computed, just return
    
    return scores
# EVOLVE-BLOCK-END
