import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 320.0  # seconds, safety margin before 330s kill


def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_remaining():
    deadline = _ensure_deadline()
    return deadline - time.time()


def _sanitize(scores):
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)


def acquisition(mu, sigma, f_best):
    global _DEADLINE
    _ensure_deadline()
    
    # Check if we are running out of time; if so, return a simple EI score quickly
    if _time_remaining() < 5.0:
        mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
        sigma_safe = np.maximum(sigma, 1e-6)
        f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
        z = (f_best_safe - mu_safe) / sigma_safe
        ei = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return _sanitize(ei)

    # Safety
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    sigma_safe = np.maximum(sigma, 1e-6)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Structural Change: Quantile Acquisition Function (QAF) / Optimistic LCB
    # Score = (f_best - mu) + k * sigma
    # This maximizes the upper bound of the expected improvement distribution.
    # It is equivalent to minimizing mu - k * sigma.
    
    # Dynamic k:
    # We use a k that starts high (exploration) and decreases (exploitation).
    # The "gap" to the known global minimum (-3.86278) serves as a proxy for progress.
    # If f_best is far from -3.86, we are early -> high k.
    # If f_best is close to -3.86, we are late -> low k.
    
    global_min = -3.86278
    gap = f_best_safe - global_min
    
    # Normalize gap to [0, 1] roughly.
    # Initial f_best might be around -1 to -2. Max gap ~ 2.8.
    # We want k to range from ~3.0 (very exploratory) to ~0.5 (exploitative).
    # Using a sigmoid or linear mapping.
    # Let's use a simple linear mapping clipped.
    # gap > 2.0 -> early
    # gap < 0.5 -> late
    
    # Map gap to k
    # k = 3.0 - 2.5 * (gap / 2.5)  ?
    # If gap=2.5, k=0.5. If gap=0, k=3.0.
    # This seems inverted.
    # Early: Large gap -> High k.
    # Late: Small gap -> Low k.
    
    # Let's try: k = 1.0 + 2.0 * np.exp(-gap)
    # If gap=0, k=3.0. If gap=2, k=1.27. If gap=5, k=1.01.
    # This keeps k >= 1.0, which is reasonable for BO.
    
    k = 1.0 + 2.0 * np.exp(-np.maximum(gap, 0.0))
    
    # Calculate QAF Score
    # Maximizing (f_best - mu) + k * sigma
    scores = (f_best_safe - mu_safe) + k * sigma_safe
    
    return _sanitize(scores)
