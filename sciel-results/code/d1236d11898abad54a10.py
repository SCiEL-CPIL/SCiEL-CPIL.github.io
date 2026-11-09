"""Rewritten program for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = None
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def _ensure_deadline():
    """Initialize or check the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME < _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a hybrid approach: primarily Expected Improvement (EI) with a
    small exploration bonus based on uncertainty, plus a slight bias toward
    the known minimum region to help convergence.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Check deadline - if we're running out of time, just return EI quickly
    if not _ensure_deadline():
        # Fallback: simple EI to be safe
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Compute a normalized uncertainty score for exploration
    # Normalize sigma to [0, 1] for a small exploration bonus
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Compute a normalized improvement score
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Hybrid score: primarily EI, with a small exploration component
    # The exploration bonus helps escape local optima in the search
    exploration_weight = 0.1
    scores = (1.0 - exploration_weight) * ei_norm + exploration_weight * sigma_norm
    
    # Add a small bias toward points with lower mu (closer to the minimum)
    # This helps focus search in promising regions
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range > 1e-12:
        # Lower mu is better, so invert
        mu_norm = (mu_max - mu) / mu_range
    else:
        mu_norm = np.zeros_like(mu)
    
    # Combine: EI dominates, with small exploration and exploitation biases
    final_scores = 0.85 * ei_norm + 0.05 * sigma_norm + 0.10 * mu_norm
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
