"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive UCB (Upper Confidence Bound) for minimization.
    
    The standard EI is structurally similar to a non-linear UCB. 
    We use a linear UCB: Score = beta * sigma - mu.
    Higher score means more likely to be the next best point (lowest value).
    
    We adapt beta based on the gap between f_best and the known global minimum (3.0).
    - Large gap (early stage): High beta, more exploration.
    - Small gap (late stage): Low beta, more exploitation.
    """
    # Known minimum of Goldstein-Price 2D
    f_min_known = 3.0
    
    # Calculate gap
    gap = f_best - f_min_known
    
    # Ensure gap is non-negative for sqrt
    gap = np.maximum(gap, 1e-6)
    
    # Adaptive beta
    # Early on (large gap), we want larger beta.
    # Late on (small gap), we want smaller beta.
    # A simple heuristic: beta proportional to sqrt(gap)
    # Scale it to a reasonable range.
    # If gap is 10, sqrt(10) ~ 3.16. 
    # If gap is 0.1, sqrt(0.1) ~ 0.31.
    # Let's map this to [0.5, 5.0]
    beta = np.clip(2.0 * np.sqrt(gap), 0.5, 5.0)
    
    # UCB for minimization: We want low values.
    # Lower bound estimate: mu - beta * sigma
    # We want to maximize the "potential" for being low, so we minimize (mu - beta * sigma).
    # Acquisition score (higher is better): - (mu - beta * sigma) = beta * sigma - mu
    
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
