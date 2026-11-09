"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement and a diversity/exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points (in objective units).
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where f_best <= mu (no improvement expected)
    # In these cases EI should be 0 or negative, but we want to focus on exploration
    # Actually, EI formula handles this: if mu >= f_best, z <= 0, 
    # (f_best - mu) is negative, Phi(z) is small, so EI is small/negative
    
    # Add a diversity bonus based on uncertainty (sigma)
    # This encourages exploring regions with high uncertainty
    # Weight the uncertainty to balance exploitation and exploration
    uncertainty_bonus = sigma_safe
    
    # Combine EI and uncertainty
    # Use a weighted combination
    # The weight on uncertainty can be tuned; here we use a moderate weight
    # to encourage exploration in uncertain regions
    
    # Normalize EI to have comparable scale with uncertainty
    # Use the max absolute value of EI for scaling
    ei_max = np.max(np.abs(ei))
    if ei_max > 1e-9:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    # Normalize uncertainty
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_normalized = sigma_safe / sigma_max
    else:
        sigma_normalized = sigma_safe
    
    # Combine with weights: 0.8 for exploitation (EI), 0.2 for exploration (uncertainty)
    # This favors finding improvements while still exploring uncertain regions
    scores = 0.8 * ei_normalized + 0.2 * sigma_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small constant to ensure all scores are distinct enough for ranking
    # This helps avoid ties that could lead to arbitrary choices
    scores += np.linspace(0, 1e-6, len(scores))
    
    return scores
# EVOLVE-BLOCK-END
