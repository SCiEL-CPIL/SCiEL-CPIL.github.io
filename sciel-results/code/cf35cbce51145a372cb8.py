"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_TIME_LIMIT = 330  # seconds


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PoI) with sigma-weighted exploration for minimization.
    Uses a time-based deadline guard. PoI favors broad coverage of multiple basins
    over EI's exploitation bias, with a small exploration term for diversity.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Deadline guard: if we are close to the time limit, return a safe,
    # exploitation-focused score to ensure the loop can finish cleanly.
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 10:
        # Fallback: simple UCB with small exploration to avoid division by zero issues
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 0.01 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This favors points where there's a good chance of beating the current best
    # without being overly biased toward exploitation like EI.
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Add a small sigma-weighted exploration term to encourage diversity
    # and help discover new basins, especially in early iterations
    # The coefficient is small to not dominate the probability signal
    exploration = 0.05 * sigma_safe
    
    scores = poi + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
