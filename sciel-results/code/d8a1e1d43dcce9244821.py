"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a blend of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration weighting based on posterior uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    t0 = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Adaptive exploration: when average uncertainty is high relative to f_best,
    # increase UCB weight to encourage broader search
    avg_sigma = np.mean(sigma_safe)
    # Normalize: if avg_sigma is large compared to f_best, we're early/exploratory
    # Use a soft transition: weight shifts from 0.7 to 0.5 EI as uncertainty grows
    # f_best starts high and decreases; sigma is in objective units
    # Scale factor: when avg_sigma > f_best * 0.1, start increasing exploration
    exploration_factor = np.clip(avg_sigma / (abs(f_best) + 1e-9), 0.0, 2.0)
    # Base alpha is 0.7 (EI weight), decrease to 0.5 max when exploration_factor is high
    alpha = 0.7 - 0.2 * np.clip(exploration_factor, 0.0, 1.0)
    
    # Combine EI and UCB with weighted average
    scores = alpha * ei + (1 - alpha) * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we've spent too much time, return immediately
    # (This function is called 25 times per seed, so each call should be fast)
    elapsed = time.time() - t0
    # 330s total / 3 seeds / 25 calls = ~4.4s per call max, but we expect <1ms
    # No need to truncate; just ensure we return
    return scores
# EVOLVE-BLOCK-END
