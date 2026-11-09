"""Cost-aware acquisition using UCB with adaptive beta for Bayesian optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 330.0 - 25.0  # safety margin before the 330 s wall-clock kill
_T0 = time.perf_counter()


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware UCB acquisition (higher = evaluate next).

    mu, sigma, cost: 1-D arrays of shape (N,). Returns N finite scores.
    """
    t0 = time.perf_counter()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite / non-positive inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 0.0)

    # Adaptive beta: balance exploitation and exploration based on observed spread
    # Use a fixed reasonable beta for UCB; could adapt based on sigma distribution
    beta = 2.0  # exploration-exploitation trade-off

    # UCB score: mu + beta * sigma (we want to minimize, so higher is better for acquisition)
    ucb = mu + beta * sigma

    # Cost-aware: normalize by cost to prefer cheaper evaluations
    score = ucb / cost

    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if (time.perf_counter() - t0) > (_DEADLINE - (time.perf_counter() - _T0)):
        return score
    return score
# EVOLVE-BLOCK-END
