"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Start time for deadline guard
    t_start = time.time()
    
    # Upper Confidence Bound (UCB) for minimization.
    # Score = -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    # For multimodal functions like Weierstrass, exploration is crucial.
    beta = 2.0
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute scores
    scores = -mu + beta * sigma_safe
    
    # Check time limit
    # The evaluator kills after 330s. We are called within a loop of 25 steps (30 evals - 5 initial).
    # Each call should be fast. UCB is O(N). N=2000. This is very fast.
    # No complex loops or heavy sampling needed.
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
