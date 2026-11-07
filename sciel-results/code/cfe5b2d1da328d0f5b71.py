"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Upper Confidence Bound (UCB) which provides a clear
    exploitation-exploration trade-off via the beta parameter.
    UCB is often more robust than EI in multimodal landscapes
    like Branin-2D because it explicitly rewards high uncertainty
    regions, helping to escape local minima.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # UCB for minimization:
    # We want to minimize f(x).
    # High score means "more promising".
    # Promising = low predicted mean (exploitation) AND high uncertainty (exploration).
    # score = -mu + beta * sigma
    
    # Determine beta.
    # A fixed beta is often sufficient for short-run BO.
    # beta = 1.0 is a standard starting point for standardized/scaled problems.
    # Given the domain of Branin (values ~0.4 to ~300), and that sigma shrinks,
    # a beta of 1.0 encourages exploration when sigma is comparable to the 
    # improvement potential.
    
    beta = 1.0
    
    # Calculate UCB scores
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
