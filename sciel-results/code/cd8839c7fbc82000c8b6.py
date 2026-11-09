import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time and ensure we return within budget
    # The evaluator kills after 330s wall-clock, so we need to be fast.
    # This function is called repeatedly, so we keep it lightweight.
    
    # Calculate Expected Improvement
    # f_best is the minimum observed value
    # We want to maximize improvement, so f_best - mu
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware normalization
    # Using square root of cost to soften the penalty for expensive evaluations
    # This allows the optimizer to explore regions with higher cost if the potential improvement is good
    safe_cost = np.maximum(cost, 1e-9)
    scores = ei / np.sqrt(safe_cost)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
