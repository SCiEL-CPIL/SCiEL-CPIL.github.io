"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    # The evaluator kills after 330s. We don't know the exact start time, 
    # but we can assume we have plenty of time for 2000 ops.
    # However, to be safe, we can just compute directly.
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Score = -mu + kappa * sigma
    # Higher score means lower expected value with a bonus for uncertainty.
    
    kappa = 2.57
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    # If sigma is nan or inf, handle it?
    # The prompt says mu and sigma are finite.
    
    return scores
# EVOLVE-BLOCK-END
