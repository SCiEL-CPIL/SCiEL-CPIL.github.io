"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Uses a Thompson Sampling proxy based on normalized improvement:
    (f_best - mu) / (sigma + eps). This encourages sampling points with
    high improvement potential relative to their uncertainty, providing
    a different exploration-exploitation balance than fixed linear
    combinations of EI and exploration terms.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # Thompson Sampling proxy: normalized improvement heuristic
    # Points with high (f_best - mu) and low sigma get high scores
    ts_proxy = (f_best - mu) / (sigma + _EPS)

    # Ensure finiteness
    scores = np.where(np.isfinite(ts_proxy), ts_proxy, 0.0)

    # Deadline guard: if we've spent too long, return the same scores
    # (they're already computed) so the caller gets valid scores.
    if time.time() - _START_TIME > _TIME_LIMIT:
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
