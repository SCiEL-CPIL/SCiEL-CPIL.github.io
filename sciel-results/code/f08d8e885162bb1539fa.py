"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and
    Upper Confidence Bound (UCB) with a time-decaying exploration coefficient.
    
    Score = w * EI + (1 - w) * (-mu + beta(t) * sigma)
    
    where w increases over time to shift from exploration to exploitation,
    and beta(t) decreases over the 30-evaluation budget.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    # Track elapsed time and call count for adaptive behavior
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    acquisition._call_count += 1
    t = acquisition._call_count  # 1-based index of this call
    
    # Total budget: 30 evaluations, 5 initial + 25 acquisition calls
    total_calls = 25
    t_clamped = min(t, total_calls)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Expected Improvement (EI) for minimization ---
    # EI = E[max(f_best - f, 0)] for f ~ N(mu, sigma^2)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- UCB component ---
    # Time-decaying exploration coefficient
    # beta(t) = 2.0 * (1 - t/30), clamped to [0.1, 2.0]
    beta = 2.0 * (1.0 - t_clamped / 30.0)
    beta = max(beta, 0.1)
    
    # UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # --- Weighting: shift from UCB to EI over time ---
    # Early: more UCB (exploration), Late: more EI (exploitation)
    # w goes from 0.3 at t=1 to 0.9 at t=25
    w = 0.3 + 0.6 * (t_clamped / total_calls)
    
    # Normalize EI and UCB to comparable scales before combining
    # Use robust scaling: divide by (max - min + eps) to avoid issues with constant arrays
    def robust_normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        range_val = arr_max - arr_min
        if range_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr_min) / range_val
    
    ei_norm = robust_normalize(ei)
    ucb_norm = robust_normalize(ucb_score)
    
    # Combine: higher weight on EI as we progress
    scores = w * ei_norm + (1.0 - w) * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
