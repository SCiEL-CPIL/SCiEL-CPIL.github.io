"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score: -mu + beta * sigma
    # Higher score means "evaluate next".
    # Using beta=2.0 to encourage stronger exploration than the beta=1.0 baseline.
    beta = 2.0
    
    # Handle case where sigma is very small to avoid numerical issues, though sigma is std dev so >= 0.
    # The formula is linear in sigma, so no division by sigma is needed.
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If mu or sigma contain inf/nan, replace them.
    # The problem statement says mu and sigma are computed by GP, so they should be finite.
    # But just in case.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
