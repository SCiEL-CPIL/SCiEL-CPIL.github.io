"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid of Expected Improvement (EI) and a small uncertainty bonus.
    EI is theoretically optimal for GP-based BO and naturally balances
    exploitation (points with low mu) and exploration (points with high sigma).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma, Phi is standard normal CDF, phi is PDF
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape flat regions where EI might be very small
    # but there's still significant uncertainty
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        exploration_bonus = 0.02 * (sigma_safe / sigma_max)
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    # Combine EI with small exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
