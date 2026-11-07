"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses a combination of Expected Improvement and a small exploration bonus
    to balance exploitation and exploration.
    
    Args:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std in objective units
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term based on uncertainty
    # This helps escape local optima early in the search
    # Scale the exploration bonus to be small relative to typical EI values
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI and exploration
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
