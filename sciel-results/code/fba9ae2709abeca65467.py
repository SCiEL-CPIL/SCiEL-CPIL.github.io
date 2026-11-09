# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for time tracking
_start_time = None
_DEADLINE_MARGIN = 15.0  # seconds before the 330s limit
_call_count = 0

def _get_elapsed():
    """Return elapsed time since module load, initializing on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded():
    """Check if we are close to the 330s wall-clock limit."""
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and 
    adaptive Upper Confidence Bound (UCB) for Bayesian optimization.
    
    This approach uses:
    1. EI for exploitation (finding points likely to improve upon f_best)
    2. Adaptive UCB for exploration (visiting uncertain regions)
    3. A time-aware fallback for safety
    
    The blend weight shifts from exploration-heavy early on to exploitation-heavy 
    as the optimization progresses (tracked by call count).
    
    Score to maximize: w_ei * EI_norm + w_ucb * UCB_norm
    """
    global _call_count
    
    # Track call count for adaptive behavior
    _call_count += 1
    
    # Ensure sigma is positive for numerical stability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Deadline guard: if we're close to the limit, use a fast fallback
    if _is_deadline_exceeded():
        # Pure exploitation: pick the point with lowest predicted mean
        return -mu
    
    # Compute Expected Improvement (EI)
    # For minimization: improvement = f_best - mu(x)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)  # Numerical stability
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Compute adaptive UCB
    # Start with higher exploration (beta ~ 3.0) and decay to lower (beta ~ 0.5)
    # Use a smooth decay based on call count
    beta = 0.5 + 2.5 * np.exp(-_call_count / 20.0)
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # Use robust scaling based on the data distribution
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + eps
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + eps
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Adaptive weighting: start with more exploration, shift to exploitation
    # Early on (few calls): w_ucb high, w_ei low
    # Later (many calls): w_ei high, w_ucb low
    # Use a smooth transition based on call count
    # At call 0: w_ucb ~ 0.6, w_ei ~ 0.4
    # At call 25: w_ucb ~ 0.2, w_ei ~ 0.8
    w_ucb = 0.2 + 0.4 * np.exp(-_call_count / 15.0)
    w_ei = 1.0 - w_ucb
    
    # Combined score
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all are -inf, return zeros as fallback
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
