"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB: Upper Confidence Bound for minimization
    # We want to maximize the score.
    # The quantity we want to minimize is f(x).
    # A point is attractive if -mu is large (low predicted f) or sigma is large (high uncertainty).
    # UCB for minimization: -mu + beta * sigma
    
    beta = 3.0
    
    # Calculate UCB score
    ucb = -mu + beta * sigma
    
    # Cost-aware normalization
    # cost is >= 1.0
    # We divide by cost to prefer cheaper points if scores are comparable.
    scores = ucb / cost
    
    # Ensure finite values
    # If sigma is 0 and mu is inf (shouldn't happen), handle it.
    # np.where can help if there are issues, but standard ops should be fine.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
