"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Use Lower Confidence Bound (LCB) for minimization.
    Score = -mu + beta * sigma.
    Higher score indicates more promising points (low mean, high uncertainty).
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls the exploration-exploitation tradeoff.
    # A value of 2.0 is a common heuristic for global optimization.
    beta = 2.0
    
    # For minimization, we want low mu and high sigma.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x).
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
