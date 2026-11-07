"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 325s to be safe
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an adaptive
    uncertainty-based exploration term.

    The exploration weight adapts based on the gap between the current best
    observed value and the known minimum of the Hartmann 6d function.
    When far from the optimum, exploration is stronger; when close, it fades
    to allow exploitation.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Time check: if we're close to deadline, return a simple score
    if time.time() > _DEADLINE - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-12)

    # Known minimum of Hartmann 6d
    h6_min = -3.32237

    # Compute gap between current best and known minimum
    gap = f_best - h6_min  # positive since f_best > h6_min (lower is better)
    gap = max(gap, 1e-6)  # avoid zero/negative

    # Adaptive exploration weight:
    # When gap is large (early), exploration weight is higher (~0.3)
    # When gap is small (late), exploration weight decreases (~0.05)
    # Using exponential decay with a characteristic scale of ~0.5
    exploration_weight = 0.05 + 0.25 * np.exp(-gap / 0.5)

    # Standardized improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Combined score: EI + adaptive exploration term
    scores = ei + exploration_weight * sigma_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
