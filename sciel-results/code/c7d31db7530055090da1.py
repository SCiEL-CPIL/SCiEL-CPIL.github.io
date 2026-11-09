"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Uses Thompson Sampling proxy: normalize improvement by uncertainty.
    Score = (f_best - mu) / (sigma + eps)
    This encourages exploration in uncertain regions while exploiting
    high-potential areas.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # Thompson Sampling proxy: normalized improvement
    scores = (f_best - mu) / (sigma + _EPS)

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if time.time() > _DEADLINE:
        z = (f_best - mu) / sigma
        z = np.clip(z, -10.0, 10.0)
        scores = np.maximum((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
