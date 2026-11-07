# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 280.0  # safety margin before the 300 s limit


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    if t0 > _DEADLINE:
        # Return a finite, benign fallback immediately.
        return np.zeros_like(mu)

    # Guard against non-finite inputs (defensive).
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Ensure finiteness.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma = np.maximum(sigma, 1e-9)

    # Cost-scaled lower confidence bound.
    # Exploration term scaled by 1/cost: high-cost regions get a smaller
    # uncertainty bonus, so we prefer exploring cheap, uncertain areas.
    # beta is a small constant; the mean term is invariant to cost.
    beta = 0.1
    exploration = beta * sigma / cost
    exploitation = mu

    score = -exploitation + exploration

    # Blend: keep it finite.
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
