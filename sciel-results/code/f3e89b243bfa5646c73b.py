# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a dynamic combination of Expected Improvement (EI) and Uncertainty (sigma).
    The exploration weight adapts based on the current best value (f_best).
    High f_best -> High exploration (to find better regions).
    Low f_best -> Low exploration (to exploit the current best region).
    """
    # Deadline check: if we're too close to the limit, return a simple score
    if _check_deadline():
        # Fallback: pure EI with a small exploration bonus
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei + 0.1 * sigma_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic exploration weight
    # Weierstrass on [-0.5, 0.5] has max value ~1.5, min 0.
    # Normalize f_best to [0, 1] range approximately
    # Clip to avoid negative weights or excessive exploration if f_best is unexpectedly high
    f_best_norm = np.clip(f_best / 1.5, 0.0, 1.0)
    
    # Linear interpolation for weight
    # When f_best is 1.5 (bad), weight is 1.5 (high exploration)
    # When f_best is 0.0 (good), weight is 0.2 (low exploration, high exploitation)
    w_explore = 0.2 + 1.3 * f_best_norm
    
    # Combine EI and Uncertainty
    score = ei + w_explore * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
