import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 300.0  # seconds
_call_count = 0
_TOTAL_STEPS = 25  # 30 total evals - 5 initial

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a dynamic hybrid of LCB (exploration) and EI (exploitation)
    based on the progress of the optimization budget.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        # Fallback: aggressive exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 2.0 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Determine progress p in [0, 1]
    # _call_count is 0 for the first acquisition call (after 5 initial points)
    # It goes up to 24 for the 25th call.
    p = _call_count / max(1, _TOTAL_STEPS - 1)
    # Clamp p to [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # 1. Calculate LCB Score (Exploration)
    # For minimization, we want to minimize mu - kappa * sigma
    # So the score (higher is better) is kappa * sigma - mu
    # We use kappa = 1.5 for strong exploration early on
    kappa = 1.5
    lcb_score = kappa * sigma_safe - mu
    
    # 2. Calculate EI Score (Exploitation)
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei_score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 3. Normalize scores to have zero mean and unit variance
    # This allows linear interpolation to be meaningful
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std < 1e-9:
        lcb_std = 1.0
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score)
    if ei_std < 1e-9:
        ei_std = 1.0
    ei_norm = (ei_score - ei_mean) / ei_std
    
    # 4. Hybrid Score
    # Blend: (1-p) * LCB + p * EI
    # Early (p~0): Mostly LCB (Explore)
    # Late (p~1): Mostly EI (Exploit)
    scores = (1.0 - p) * lcb_norm + p * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Increment call count
    _call_count += 1
    
    return scores
