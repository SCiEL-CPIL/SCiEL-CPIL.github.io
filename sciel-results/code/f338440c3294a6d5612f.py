# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 315.0

_call_count = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _call_count, _last_f_best
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei + 1.5 * sigma_safe
    
    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta
    mean_sigma = np.mean(sigma_safe)
    std_sigma = np.std(sigma_safe)
    if mean_sigma > 1e-12:
        # beta decreases as we progress, encouraging exploitation
        beta = 2.0 * np.exp(-0.15 * _call_count) + 0.5
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize components
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # Adaptive weighting: start with more exploration, shift to exploitation
    # frac_better measures how many points already beat f_best
    frac_better = np.mean(mu < f_best)
    
    # Early steps: more UCB (exploration), later: more EI (exploitation)
    # Use a smooth transition
    progress = min(_call_count / 25.0, 1.0)  # 30 total evals, 5 initial
    w_ucb = 0.6 * (1.0 - progress) + 0.1
    w_ei = 0.4 * progress + 0.3
    
    # Boost exploitation if many points are already better
    w_ei += 0.2 * frac_better
    w_ucb *= (1.0 - 0.3 * frac_better)
    
    # Ensure weights sum to 1
    total_w = w_ei + w_ucb
    w_ei /= total_w
    w_ucb /= total_w
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
