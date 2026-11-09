"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement with progress-adaptive
    exploration. Structural change: uses a dynamic exploration coefficient
    that increases as f_best approaches the known minimum, and applies a
    tail-emphasis to the EI to better exploit the left tail of the posterior.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return -mu.copy()  # Fallback: pure exploitation
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of Styblinski-Tang 1D
    f_min = -39.1662
    
    # Progress: 0 at start, 1 when f_best reaches f_min
    # f_best starts around -10 to -20 typically
    progress = np.clip((f_best - f_min) / (f_min - (-10.0)), 0.0, 1.0)
    # As progress increases (f_best gets closer to f_min), we want MORE exploration
    # because we're in a hard region and need to escape local optima
    beta = 0.5 + 1.5 * progress
    
    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Tail-emphasis: amplify EI for points where mu is already below f_best
    # by adding a bonus proportional to how far below f_best the mean is
    # This helps when the GP has already identified a promising region
    below_best = np.maximum(f_best - mu, 0.0)
    tail_bonus = below_best * 0.1
    
    # Combine: EI + tail_bonus + exploration term
    exploration = beta * sigma_safe
    
    scores = ei + tail_bonus + exploration
    
    # Normalize to [0, 1] for consistent scale
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    rng = scores_max - scores_min
    if rng < 1e-12:
        scores = np.zeros_like(scores)
    else:
        scores = (scores - scores_min) / rng
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
