"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.array): Mean of the Gaussian process posterior at candidate points.
    sigma (np.array): Standard deviation of the Gaussian process posterior at candidate points.
    f_best (float): The best (lowest) observed value so far.
    
    Returns:
    np.array: Scores for each candidate point. Higher score indicates better candidate.
    """
    # UCB for minimization:
    # We want to minimize f(x).
    # Uncertainty is high when sigma is high.
    # We prefer low mu and high sigma.
    # Score = beta * sigma - mu
    # Higher score means better candidate (more promising or more uncertain).
    
    # Clip sigma to avoid numerical issues, though it should be positive
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Exploration-exploitation trade-off parameter
    # A value of 3.0 is a common heuristic for early-mid optimization
    beta = 3.0
    
    ucb = beta * sigma_safe - mu
    
    # Ensure finite scores
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
# EVOLVE-BLOCK-END
