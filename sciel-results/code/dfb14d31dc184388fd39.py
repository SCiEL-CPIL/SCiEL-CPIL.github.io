"""Rewritten program using UCB with adaptive exploration and noise-aware safety."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred for evaluation.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter beta
    # Start with a moderate value and adjust based on the relationship between sigma and mu
    # For minimization, we want to explore where uncertainty is high relative to the best value
    
    # Compute a baseline exploration coefficient
    # The scale of sigma relative to the range of mu helps calibrate exploration
    mu_range = np.ptp(mu) if len(mu) > 1 else 1.0
    if mu_range < 1e-9:
        mu_range = 1.0
    
    # Base beta: balance between exploitation and exploration
    # A typical UCB for minimization: -mu + beta * sigma
    # We want higher scores for lower mu and higher sigma
    
    # Adaptive beta: increase exploration when sigma is large relative to the scale of the problem
    # Use a sigmoid-like adaptation based on the median sigma
    sigma_median = np.median(sigma_safe)
    mu_scale = max(mu_range, 1e-9)
    
    # Normalize sigma by the scale of mu to get a relative uncertainty measure
    relative_sigma = sigma_median / mu_scale
    
    # beta increases with relative uncertainty, capped to avoid excessive exploration
    # At low uncertainty, beta is small (more exploitation)
    # At high uncertainty, beta is larger (more exploration)
    base_beta = 2.0
    # Adapt beta: if relative sigma is large, increase beta; if small, decrease it
    adaptive_factor = 1.0 + 0.5 * np.log1p(relative_sigma * 10)
    beta = base_beta * adaptive_factor
    
    # Clamp beta to reasonable bounds
    beta = np.clip(beta, 0.5, 5.0)
    
    # Compute UCB scores for minimization
    # We want to maximize: -mu + beta * sigma
    # This favors points with low predicted mean and high uncertainty
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
