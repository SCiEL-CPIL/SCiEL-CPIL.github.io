"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = None
_START_TIME = None


def _get_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        # The evaluator kills after 330s; use a safety margin
        _DEADLINE = _START_TIME + 320.0
    return _DEADLINE


def _time_remaining():
    return _get_deadline() - time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a blend of Expected Improvement (EI) and GP-LCB
    (Lower Confidence Bound) for minimization.
    
    mu, sigma: shape (2000,), posterior mean and std at candidate points.
    f_best: lowest observed value so far.
    
    Returns: shape (2000,) scores, higher = evaluate next.
    """
    # Time guard: if we're running out of time, return a simple but fast score
    if _time_remaining() < 2.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    n = len(mu)
    eps = 1e-9

    # --- Core: Expected Improvement for minimization ---
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # --- GP-LCB component for minimization ---
    # LCB(x) = mu(x) - kappa * sigma(x)
    # Lower LCB means more promising (potentially lower function value).
    # Score = -LCB = -mu + kappa * sigma (higher is better)
    kappa = 1.5
    lcb_score = -mu + kappa * sigma

    # --- Smooth blend without normalization ---
    # Both EI and lcb_score are in "higher is better" units.
    # EI is bounded below by 0; lcb_score can be negative.
    # We blend them with a weight that shifts from exploration to exploitation.
    
    # Estimate how close we are to the optimum using the posterior scale.
    # A robust measure: the fraction of candidates where EI is meaningfully > 0
    # (i.e., where the posterior suggests improvement over f_best).
    # This adapts naturally as f_best improves.
    ei_positive_frac = np.mean(ei > 0.01 * (np.max(ei) + eps))
    
    # Also use the ratio of f_best to the typical function scale
    mu_scale = np.median(np.abs(mu) + sigma)
    if mu_scale < 1e-9:
        mu_scale = 1.0
    
    # improvement_ratio: 0 when f_best is near typical values (early), 
    # 1 when f_best is much lower (late, close to optimum)
    improvement_ratio = max(0.0, min(1.0, (mu_scale - f_best) / (mu_scale + eps)))
    
    # Weight: early on, rely more on LCB for exploration; late, rely more on EI
    # w_ei goes from ~0.4 (early) to ~0.85 (late)
    w_ei = 0.4 + 0.45 * improvement_ratio
    w_lcb = 1.0 - w_ei

    # Scale lcb_score to be comparable to EI
    # EI is typically O(sigma) in magnitude, lcb_score is O(mu + sigma)
    # Normalize lcb_score by its own range to make it O(1)
    lcb_min, lcb_max = np.min(lcb_score), np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + eps
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # EI is already non-negative and naturally scaled; normalize it too for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei - ei_min) / ei_range

    score = w_ei * ei_norm + w_lcb * lcb_norm

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
