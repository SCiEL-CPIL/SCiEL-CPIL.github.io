"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    This approach balances exploitation (low mean) and exploration (high uncertainty)
    linearly.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, the Lower Confidence Bound is:
    # LCB = mu - beta * sigma
    # We want to select the point with the lowest LCB.
    # The acquisition function should return higher values for more promising points.
    # Therefore, score = -LCB = beta * sigma - mu
    
    # Tuning parameter beta:
    # Higher beta encourages more exploration.
    # For 1D Weierstrass with length scale 0.2, beta=2.5 is a reasonable balance.
    beta = 2.5
    
    score = beta * sigma - mu
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
