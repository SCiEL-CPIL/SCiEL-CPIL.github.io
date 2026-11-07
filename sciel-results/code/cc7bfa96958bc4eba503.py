"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a global start time so we can bail out safely.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Deterministic GP-based acquisition: maximize expected improvement
    with a small exploration bonus derived from the predictive std.
    
    mu: (N,) predictive mean
    sigma: (N,) predictive std
    f_best: float, best observed value (lower is better)
    
    Returns: (N,) scores, higher = more promising.
    """
    # Guard against deadline
    if _check_deadline():
        # Return a deterministic fallback: just the negative mean
        return -mu.copy()
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus: prefer points with higher uncertainty
    # when the EI is close to zero (i.e., the point is near the current best).
    # This helps escape local optima in the acquisition landscape.
    # Scale the bonus so it's small relative to typical EI values.
    sigma_scale = np.max(sigma_safe) if np.max(sigma_safe) > 1e-12 else 1.0
    exploration = 0.05 * (sigma_safe / sigma_scale)
    
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
