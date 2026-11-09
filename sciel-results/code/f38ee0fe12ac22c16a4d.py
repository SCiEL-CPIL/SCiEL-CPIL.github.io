"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a Lower Confidence Bound (LCB) strategy for minimization.
    Score = -mu + beta * sigma
    
    This encourages exploration of high-uncertainty regions (high sigma)
    while still favoring regions with low predicted means (low mu).
    This is often more robust for multimodal functions like Michalewicz 2D
    than Expected Improvement, which can be overly greedy.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid issues
    sigma = np.maximum(sigma, 1e-12)
    
    # LCB for minimization: we want to maximize -mu + beta * sigma
    # beta controls the exploration/exploitation trade-off
    beta = 2.5
    
    # Calculate score
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline (unlikely to be hit for such simple computation, but good practice)
    if time.time() > DEADLINE:
         return score

    return score
# EVOLVE-BLOCK-END
