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
    Uses high static sigma-inflated EI with a constant LCB bonus to aggressively
    target the narrow minimum basin of the Beale function.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)

    # High static sigma inflation: maintains a "hilly" EI landscape even when
    # posterior uncertainty is low, preventing premature convergence to flat regions.
    inflation = 4.0
    sigma_eff = sigma * inflation
    sigma_safe = np.where(sigma_eff > 1e-9, sigma_eff, 1e-9)

    # Compute sigma-inflated Expected Improvement
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent extreme values in norm.cdf/pdf which can lead to numerical noise
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Constant LCB bonus: provides persistent exploration pressure.
    # The Beale function requires sustained exploration to find the narrow basin.
    beta = 2.5
    lcb_bonus = beta * sigma

    # Combine: EI (primary) + LCB (exploration)
    score = ei + lcb_bonus

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
