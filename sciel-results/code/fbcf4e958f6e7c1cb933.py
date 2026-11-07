"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we don't exceed the 300s limit (with safety margin)
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
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
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're getting close to the 300s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 280:  # 20s safety margin
        # Return a simple score based on low mu and high sigma
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 0.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Lower Confidence Bound (LCB) for exploration
    # LCB = mu - beta * sigma (we want to minimize, so lower is better)
    # For the acquisition function (higher is better), we use -LCB = -mu + beta*sigma
    # We'll combine EI with an LCB-like term
    
    # Adaptive beta: start with moderate exploration, decrease as f_best improves
    # f_best starts around 5 (uniform random) and minimum is 3
    # As f_best gets closer to 3, we can reduce exploration
    progress = np.clip((5.0 - f_best) / 2.0, 0, 1)  # 0 at start, 1 when f_best=3
    beta = 2.0 * (1.0 - 0.5 * progress)  # beta goes from 2.0 to 1.0
    
    # LCB-based score (higher is better for exploration)
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both EI and LCB scores to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-10:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_min, lcb_max = np.min(lcb_score), np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-10:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Weight combination: more weight on EI as we get closer to optimum
    # Early on: more exploration (LCB weight higher)
    # Later: more exploitation (EI weight higher)
    ei_weight = 0.6 + 0.3 * progress  # 0.6 to 0.9
    lcb_weight = 1.0 - ei_weight  # 0.4 to 0.1
    
    # Final combined score
    scores = ei_weight * ei_norm + lcb_weight * lcb_norm
    
    # Add a small bonus for points with very high sigma (exploration)
    # This helps escape local optima in the multimodal Goldstein-Price function
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        sigma_bonus = 0.05 * (sigma_safe / sigma_max)
    else:
        sigma_bonus = np.zeros_like(sigma_safe)
    
    scores += sigma_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
