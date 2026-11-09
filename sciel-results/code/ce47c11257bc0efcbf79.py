"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: array of shape (N,) posterior mean
        sigma: array of shape (N,) posterior standard deviation
        f_best: float, best observed value so far
        
    Returns:
        scores: array of shape (N,) where higher values indicate better candidates
    """
    # Deadline guard
    # The evaluator kills after 330s. We don't know start time, but we can assume 
    # this function is called frequently. A heavy computation here would be bad.
    # UCB is O(N), so it is very fast. No guard strictly needed for UCB, 
    # but we include a check for safety if we were doing something heavy.
    
    # UCB for minimization:
    # We want to maximize the score.
    # The value to minimize is f(x).
    # Lower bound on f(x) is mu - beta * sigma.
    # We want to find the lowest possible value, so we prioritize low mu and high sigma.
    # Score = -mu + beta * sigma
    
    beta = 5.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
