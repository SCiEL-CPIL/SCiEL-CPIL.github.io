"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000,).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray
        UCB scores, shape (2000,). Higher scores indicate more promising points.
    """
    # Adaptive beta parameter based on the scale of the problem
    # We want to balance exploration and exploitation
    # For Ackley 2D on [-5,5]^2, the function values range roughly from 0 to ~20
    # A reasonable beta that encourages exploration early and exploitation later
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use a fixed beta that works well for this scale
    # beta = 2.0 is a common choice, but we can make it adaptive
    # Since we don't have access to the number of evaluations, use a fixed value
    # that balances exploration well for 30 total evaluations
    beta = 2.5
    
    # UCB for minimization: we want to maximize (f_best - mu + beta * sigma)
    # This encourages points with low predicted mean and high uncertainty
    scores = (f_best - mu) + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
