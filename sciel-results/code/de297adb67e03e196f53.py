"""Rewritten program using Knowledge Gradient acquisition for Michalewicz 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = time.time() + 280.0  # 280s margin within 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach: Knowledge Gradient-inspired scoring that combines
    the expected improvement with a penalty for high uncertainty in regions
    where the mean is already promising. This is structurally different from
    standard EI by using a variance-aware weighting that adapts based on how
    close the mean is to f_best.
    
    The score is:
      score = EI * (1 + alpha * exp(-|z|))
    where z = (f_best - mu) / sigma, EI is standard expected improvement,
    and alpha is a tunable parameter that controls the exploration boost
    for uncertain regions.
    
    This creates a different search landscape than pure EI: it upweights
    exploration when uncertainty is high relative to the potential improvement,
    which can help escape local optima in the Michalewicz function.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Return a simple fallback if we're running out of time
        return np.maximum(f_best - mu, 0)
    
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard EI for minimization
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Knowledge Gradient-inspired modulation:
    # Boost exploration when uncertainty is high (large |z| means either
    # mu is far below or above f_best relative to sigma)
    # Use a Gaussian kernel on |z| to create smooth modulation
    alpha = 0.5  # Exploration boost parameter
    exploration_boost = 1.0 + alpha * np.exp(-np.abs(z_clipped))
    
    # Apply the modulation
    score = ei * exploration_boost
    
    # Ensure finite values and non-negative
    score = np.maximum(score, 0)
    score = np.where(np.isfinite(score), score, 0)
    
    return score
# EVOLVE-BLOCK-END
