"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Linear Expected Improvement / Optimistic Score
    # Encourages points with low mean (high potential) and high uncertainty (exploration)
    # Distinct from EI by using a linear improvement term plus a fixed exploration bonus
    # instead of a probability-weighted improvement.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Base score: Potential improvement (f_best - mu)
    # Exploration bonus: beta * sigma
    # Beta = 1.0 is a heuristic for this scale
    beta = 1.0
    
    score = (f_best - mu) + beta * sigma
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
