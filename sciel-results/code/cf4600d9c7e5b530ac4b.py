"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Minimum-variance acquisition with a small exploitation term.
    
    The primary signal is the posterior standard deviation (uncertainty),
    which forces the GP to resolve uncertainty in unexplored regions.
    A small negative mean term provides mild exploitation to avoid
    pure random exploration while still prioritizing uncertainty.
    
    This structural change from EI to min-variance aims to prevent
    premature convergence to local noise minima in the 1D landscape.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Primary: maximize uncertainty (sigma)
    # Secondary: mild exploitation toward low mean (small weight)
    # Score = sigma - 0.01 * mu
    # Higher score means evaluate next
    
    # Use a small exploitation weight to balance exploration
    exploration_weight = 1.0
    exploitation_weight = 0.01
    
    scores = exploration_weight * sigma - exploitation_weight * mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
