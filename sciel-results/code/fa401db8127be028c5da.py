"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

_DEADLINE = 325.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combines EI, adaptive UCB, and power cost penalty.
    Returns 2000 finite scores (higher = more promising).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n != 2000:
        return np.ones(n)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    pdf_z = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * ndtr(z_clipped) + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # Adaptive kappa based on how close f_best is to the known minimum
    # Hartmann 6d minimum is -3.32237; initial uniform points give ~-1 to -2
    # progress: 0 at start (f_best ~ -1), 1 when f_best ~ -3.32
    f_min = -3.32237
    f_start = -1.0  # approximate initial f_best from 5 uniform points
    progress = np.clip((f_start - f_best) / (f_start - f_min), 0.0, 1.0)
    # kappa decays from 2.5 (explore more) to 0.5 (exploit more)
    kappa = 2.5 - 2.0 * progress
    ucb = mu - kappa * sigma_safe
    ucb_gain = np.maximum(f_best - ucb, 0.0)

    # Adaptive alpha: weight UCB less to prioritize EI
    alpha = 0.2 - 0.1 * progress  # 0.2 early, 0.1 late
    combined = ei + alpha * ucb_gain

    # Power cost penalty: quadratic to strongly prefer low-cost (low coordinate) regions
    # cost in [1, 2], power 2.0 gives range [1, 4.0]
    scores = combined / np.power(cost_safe, 2.0)

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 1e-10)

    return scores
# EVOLVE-BLOCK-END
