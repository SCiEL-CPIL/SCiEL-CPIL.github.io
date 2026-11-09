# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = None
DEADLINE_FRACTION = 0.92  # use 92% of 300s to be safe


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive Expected Improvement with
    uncertainty-aware weighting. Blends EI with a sharpness-based
    dynamic selection between exploitation and exploration.
    
    Returns 2000 finite scores; higher means evaluate next.
    """
    global _start_time
    
    # Ensure timer is initialized
    if _start_time is None:
        _start_time = time.perf_counter()
    
    elapsed = _get_elapsed()
    
    # Deadline guard: if we're running out of time, fall back to simple EI
    # to ensure we return quickly
    if elapsed > DEADLINE_FRACTION * 300.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute EI scores
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Compute a sharpness metric: how peaked the posterior is
    # Low sigma relative to f_best - mu suggests high confidence
    # Use normalized sigma to assess uncertainty
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Sharpness: high when sigma is small relative to improvement potential
    improvement_potential = f_best - np.mean(mu)
    if improvement_potential > 1e-9:
        sharpness = 1.0 / (1.0 + sigma_mean / improvement_potential)
    else:
        sharpness = 0.5
    
    # Compute UCB scores for exploration
    # Adaptive beta based on sharpness and progress
    progress = min(elapsed / 300.0, 1.0)
    
    # When sharpness is high (low uncertainty), reduce exploration
    # When sharpness is low (high uncertainty), increase exploration
    beta_base = 2.0 - 1.5 * sharpness  # range [0.5, 2.0]
    beta_time_decay = 1.0 - 0.5 * progress  # range [0.5, 1.0]
    beta = beta_base * beta_time_decay
    
    ucb_scores = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for blending
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    ei_norm = _normalize(ei_scores)
    ucb_norm = _normalize(ucb_scores)
    
    # Dynamic blend weight based on sharpness
    # High sharpness -> more EI (exploitation)
    # Low sharpness -> more UCB (exploration)
    w_ei = 0.4 + 0.4 * sharpness
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
