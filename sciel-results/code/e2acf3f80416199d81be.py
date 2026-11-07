"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array of shape (2000,) - GP mean
    sigma: array of shape (2000,) - GP standard deviation
    f_best: float - best observed value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    array of shape (2000,) - scores (higher is better)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if elapsed time exceeds 325s, return a safe fallback
    # to ensure we finish before the 330s kill switch.
    if time.time() - _start_time > 325:
        # Fallback: Greedy exploitation adjusted by cost
        # Minimize mu / cost -> Maximize -mu / cost
        return -mu / (cost + 1e-9)

    # Thompson Sampling:
    # 1. Draw a sample from the posterior distribution: f_sample = mu + sigma * N(0, 1)
    # 2. Select the point that minimizes f_sample / cost
    # 3. Return scores such that the maximum score corresponds to the minimum f_sample / cost
    
    noise = np.random.randn(len(mu))
    sampled_values = mu + sigma * noise
    
    # We want to minimize sampled_values / cost.
    # The evaluator picks the candidate with the highest score.
    # So we return the negative of the ratio.
    scores = -sampled_values / (cost + 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
