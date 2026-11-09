"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 300.0  # wall-clock budget guard (evaluator kills at 330s)


def acquisition(mu, sigma, f_best, cost):
    # Structural change: treat cost as a discount factor in a stochastic
    # shortest-path view. Use the geometric mean of normalized Expected
    # Improvement and inverse cost to balance exploitation and efficiency
    # without the instability of division.
    if time.time() > _DEADLINE:
        # Safety margin: fall back to a cheap, finite heuristic.
        return -mu

    eps = 1e-9
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Expected improvement (minimization).
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.clip(ei, 0.0, None)

    # Normalize EI to (0, 1] scale via its max, keep a floor for log.
    ei_max = np.max(ei)
    if not np.isfinite(ei_max) or ei_max <= 0.0:
        return -mu
    ei_norm = ei / ei_max
    ei_norm = np.clip(ei_norm, 1e-12, None)

    # Inverse cost normalized to (0, 1] via min cost.
    c_min = np.min(cost)
    if not np.isfinite(c_min) or c_min <= 0.0:
        c_min = 1.0
    inv_cost_norm = c_min / np.maximum(cost, eps)
    inv_cost_norm = np.clip(inv_cost_norm, 1e-12, None)

    # Geometric mean as a discounted-reward balance.
    score = np.sqrt(ei_norm * inv_cost_norm)
    return score
# EVOLVE-BLOCK-END
