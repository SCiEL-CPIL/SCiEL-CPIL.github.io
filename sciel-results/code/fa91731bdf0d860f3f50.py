import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) for minimization with adaptive exploration.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add a small exploration bonus based on sigma to encourage exploration early
    # Use a modest multiplier to balance exploitation and exploration
    exploration_bonus = 0.1 * sigma_safe
    
    # Final score: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
