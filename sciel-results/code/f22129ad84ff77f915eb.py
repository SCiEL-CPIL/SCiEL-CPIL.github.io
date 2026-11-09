# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = None
DEADLINE = 300.0
SAFETY_MARGIN = 25.0  # seconds before deadline to start returning fast


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive
    exploration/exploitation balance.
    
    Returns 2000 finite scores; higher means evaluate next.
    """
    global _start_time
    
    # Ensure timer is initialized
    if _start_time is None:
        _start_time = time.perf_counter()
    
    elapsed = _get_elapsed()
    
    # Deadline guard: if we're running out of time, fall back to simple EI
    if elapsed > DEADLINE - SAFETY_MARGIN:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Estimate progress through the optimization budget
    # Total budget is 30 evaluations, each taking some time
    # Use elapsed time as proxy: 0 at start, 1 at end
    progress = min(elapsed / (DEADLINE - SAFETY_MARGIN), 1.0)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB (Upper Confidence Bound) for exploration
    # UCB = -mu + beta * sigma (higher is better for minimization)
    # Adaptive beta: start high for exploration, decay for exploitation
    beta_start = 2.5
    beta_end = 0.8
    beta = beta_start - (beta_start - beta_end) * progress
    
    ucb_scores = -mu + beta * sigma
    
    # Normalize both to [0, 1] for blending
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    ei_norm = _normalize(ei_scores)
    ucb_norm = _normalize(ucb_scores)
    
    # Blend: start with more exploration (UCB), shift to exploitation (EI)
    # At start: w_ei = 0.4, w_ucb = 0.6
    # At end:   w_ei = 0.85, w_ucb = 0.15
    w_ei = 0.4 + 0.45 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
