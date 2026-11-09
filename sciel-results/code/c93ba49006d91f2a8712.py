"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    
    Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
    UCB encourages exploration by adding a term proportional to the uncertainty
    (sigma) to the predicted mean (mu). For minimization, we maximize:
    score = -mu + beta * sigma
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means more promising
    """
    # Ensure sigma is positive and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration-exploitation trade-off parameter
    # A value of 1.0-2.0 is typical for small to medium sample sizes.
    # We use 1.5 to encourage exploration in the early stages of the 30 evaluations.
    beta = 1.5
    
    # UCB for minimization:
    # We want to maximize the probability of finding a low value.
    # Equivalent to maximizing the upper confidence bound of the negative function.
    # score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
