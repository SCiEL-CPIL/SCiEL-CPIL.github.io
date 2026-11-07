"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    UCB balances exploration and exploitation by adding a bonus term proportional
    to the uncertainty (sigma). For minimization, we want to minimize mu - beta*sigma,
    so the score (higher is better) is beta*sigma - mu.
    
    This is structurally different from Expected Improvement: UCB is a one-step
    confidence-bound approach, while EI integrates over the improvement distribution.
    For small-budget problems on unimodal functions, UCB's explicit exploration bonus
    can help avoid premature convergence.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization: score = beta * sigma - mu
    # Higher score means more promising (either low mean or high uncertainty)
    beta = 2.0
    scores = beta * sigma_safe - mu
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 295:
        pass
    
    return scores
# EVOLVE-BLOCK-END
