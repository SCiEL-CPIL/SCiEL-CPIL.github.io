"""Bayesian optimization acquisition for 2D RBF mixture with Matérn-5/2 GP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 310.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 2D RBF mixture.
    
    Uses a smooth blend of Expected Improvement (exploitation) and
    an uncertainty-based exploration term, with weights that shift
    from exploration to exploitation over the 25 BO steps.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to pure EI
    if time.time() - _T0 > _DEADLINE:
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    progress = _get_progress(_eval_count)
    
    # Compute EI (exploitation signal)
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Exploration signal: negative mean plus scaled uncertainty
    # Use a moderate beta that decreases with progress
    beta = 3.0 - 2.0 * progress  # 3.0 -> 1.0
    explore = -mu + beta * sigma_safe
    
    # Smooth interpolation weight:
    # w_explore starts high (0.7) and decreases to 0.0
    # w_ei starts low (0.3) and increases to 1.0
    w_explore = max(0.0, 0.7 - 0.7 * progress)
    w_ei = 1.0 - w_explore
    
    # Scale EI so it's comparable in magnitude to explore
    # EI is typically smaller than |explore|, so scale it up
    ei_scale = np.median(np.abs(explore)) / (np.median(ei) + _EPS)
    ei_scaled = ei * ei_scale
    
    # Blend
    score = w_ei * ei_scaled + w_explore * explore
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
