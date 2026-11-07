"""Acquisition function for Bayesian optimization of styblinski tang 1d."""
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # 330s wall-clock limit from evaluator

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining normalized EI with inverted UCB for
    exploration-exploitation balance on multi-modal Styblinski-Tang.
    
    Parameters:
        mu: array of shape (2000,) - GP posterior mean in objective units
        sigma: array of shape (2000,) - GP posterior std in objective units
        f_best: float - lowest observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores, higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return safe scores
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return scores that favor exploitation (low mu) to be safe
        return -mu
    
    # Ensure sigma is strictly positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range roughly for scaling
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Inverted UCB: rewards high uncertainty (sigma)
    # Scale sigma to be comparable to normalized EI
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        ucb_term = sigma_safe / sigma_max
    else:
        ucb_term = sigma_safe
    
    # Combine: exploitation (EI) + exploration (uncertainty)
    # Weight: 70% exploitation, 30% exploration
    score = 0.7 * ei_norm + 0.3 * ucb_term
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
