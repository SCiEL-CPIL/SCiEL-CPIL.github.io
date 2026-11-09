"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Safety margin: 300s limit, aim for < 280s total.
# The acquisition function itself is very fast, so the deadline guard is mostly for safety.
# However, we don't have a global timer here, just the function call.
# The evaluator calls this function repeatedly.
# We assume the evaluator handles the loop termination.

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for Probability of Improvement
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher is better.
    # We want high PI and low cost.
    score = pi / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
