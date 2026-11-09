# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_TOTAL_BUDGET = 330.0
_SAFETY_MARGIN = 5.0
_CALL_COUNT = 0


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _deadline_check(safety_margin=_SAFETY_MARGIN):
    """Return True if we are safe to continue."""
    return _get_elapsed() < _TOTAL_BUDGET - safety_margin


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    Combines sigma-inflated EI with a normalized LCB component, using
    adaptive blending that shifts from exploration to exploitation.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    t = _CALL_COUNT

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)

    # Sigma inflation: moderate to keep EI landscape hilly but not overly distorted
    inflation = 2.5
    sigma_eff = sigma * inflation
    sigma_safe = np.where(sigma_eff > 1e-9, sigma_eff, 1e-9)

    # Compute sigma-inflated Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # LCB score: -mu + beta * sigma (higher is better)
    # Beta decays gently over time
    beta_start = 3.0
    beta_decay = 0.08
    beta = beta_start / (1.0 + beta_decay * t)
    lcb_score = -mu + beta * sigma_safe

    # Normalize both components to [0, 1] for meaningful blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb_score), np.max(lcb_score)

    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0

    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (lcb_score - lcb_min) / lcb_range

    # Adaptive weighting: shift from LCB-dominated to EI-dominated
    # w_lcb decays; w_ei = 1 - w_lcb
    w_lcb = np.exp(-t / 10.0)
    w_ei = 1.0 - w_lcb

    # Blend the two normalized scores
    score = w_ei * ei_norm + w_lcb * lcb_norm

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
