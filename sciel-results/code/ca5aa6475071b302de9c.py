import numpy as np
import time

# Track elapsed time to enforce the 300s per evaluation / 330s total limit
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Thompson Sampling.
    
    Parameters:
    mu: array of mean predictions (2000,)
    sigma: array of standard deviations (2000,)
    f_best: lowest observed value
    cost: array of evaluation costs (2000,)
    
    Returns:
    scores: array of acquisition scores (2000,), higher is better
    """
    # Deadline guard: if we are running out of time, fall back to a fast, deterministic strategy
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 5.0:
        # Fallback: Greedy exploitation (minimize mu / cost)
        # Using -mu/cost because we maximize score
        scores = -mu / np.maximum(cost, 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure sigma is strictly positive to avoid issues with sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a sample from the posterior for each candidate
    # This stochastically balances exploration (high sigma) and exploitation (low mu)
    samples = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # Cost-aware scoring: We want to minimize samples / cost
    # So we maximize -samples / cost
    # cost is >= 1.0, so no division by zero issues, but we guard just in case
    scores = -samples / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
