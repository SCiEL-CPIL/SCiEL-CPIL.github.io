"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Combines Expected Improvement (EI) with a small uncertainty bonus
    to balance exploitation of promising regions with exploration of
    uncertain areas, helping to find all local minima in Branin-2D.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative for numerical stability
    ei = np.maximum(ei, 0.0)
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps escape deep local minima and find other minima in Branin-2D
    # The bonus is scaled relative to the current best value to remain adaptive
    scale = max(abs(f_best), 1.0)
    uncertainty_bonus = 0.1 * sigma_safe / scale
    
    scores = ei + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
