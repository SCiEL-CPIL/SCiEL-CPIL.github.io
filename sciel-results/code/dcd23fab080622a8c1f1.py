"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture.
    
    Uses a hybrid approach combining Expected Improvement (EI) with an exploration bonus
    that adapts based on the spread of uncertainties and the current best value.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is non-negative and has a small floor to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add an exploration component that rewards high uncertainty
    # but is scaled to be meaningful relative to EI
    # Use a square root of sigma to balance exploration
    exploration_weight = 0.1
    exploration = exploration_weight * np.sqrt(sigma)
    
    # Combine EI and exploration
    # The exploration term helps escape local optima early in the search
    # while EI dominates when exploitation is more beneficial
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
