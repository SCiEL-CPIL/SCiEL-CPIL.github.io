"""Rewritten program using UCB acquisition function with dynamic beta."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization
    with dynamic exploration constant.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Dynamic beta: start aggressive (high exploration) early, 
    # shift to exploitation as f_best improves
    # Ackley 2D minimum is 0, so f_best starts positive and decreases toward 0
    # Use f_best to modulate beta: higher f_best -> more exploration
    
    # Scale: when f_best is large (early), beta is high; when f_best is small (late), beta is low
    # Use a smooth transition: beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    beta_min = 0.5
    beta_max = 2.0
    scale = 0.5
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best / scale)
    
    # For minimization: maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
