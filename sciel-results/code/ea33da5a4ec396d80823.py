"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI)
    # mu, sigma, f_best are in objective units
    # cost is the cost of evaluation at each candidate
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best
    pi = norm.cdf(z)
    
    # Exponential cost penalty
    # Cost is typically in [1, 2] for this task
    # We penalize higher costs exponentially to favor cheap evaluations
    # Lambda controls the strength of the cost preference
    lambda_cost = 2.0
    penalty = np.exp(-lambda_cost * cost)
    
    # Acquisition score: Higher is better
    scores = pi * penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
