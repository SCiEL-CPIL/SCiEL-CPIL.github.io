import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_LIMIT = 300.0
_EVAL_COUNT = 0

def _min_max_normalize(arr):
    """Normalize array to [0, 1] range safely."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min
    if arr_range < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - arr_min) / arr_range

def acquisition(mu, sigma, f_best):
    global _EVAL_COUNT
    _EVAL_COUNT += 1
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _LIMIT - 5.0:
        return -mu

    # Ensure inputs are finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # We want to maximize score, so we use -mu + beta*sigma
    # Adapt beta based on evaluation count: more exploration early, less later
    # Total evaluations per seed: 30, so normalize by 30
    progress = min(_EVAL_COUNT / 30.0, 1.0)
    # Start with high exploration (beta=3.0), decrease to low exploitation (beta=0.5)
    beta = 3.0 - 2.5 * progress
    lcb = -mu + beta * sigma
    
    # Normalize scores
    ei_norm = _min_max_normalize(ei)
    lcb_norm = _min_max_normalize(lcb)
    
    # Adaptive weighting: more EI early when we have less info, more LCB later
    # Actually, we want more exploration early, so weight LCB more early
    w_ei = 0.3 + 0.4 * progress  # 0.3 early, 0.7 late
    w_lcb = 1.0 - w_ei
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
