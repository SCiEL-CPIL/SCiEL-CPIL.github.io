"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses a robust combination of Expected Improvement (EI) and a small
    exploration bonus based on uncertainty, designed to balance exploitation
    and exploration for the 1D sphere function on [-5, 5].
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical edge cases
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # The bonus is scaled to be small relative to typical EI values
    exploration_bonus = 0.01 * sigma_safe
    
    # Combine: primarily EI, with slight exploration encouragement
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
