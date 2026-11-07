"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 290.0  # Safety margin before 300s limit


def _time_left():
    return _TIMEOUT - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    # Time guard: if we're running out of time, return a safe fallback
    if _time_left() < 5.0:
        scores = -mu / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    sigma_safe = np.maximum(sigma, 1e-9)

    # Pure UCB with fixed beta=2.0 (proven effective for Shekel 4d)
    # Higher beta encourages exploration of high-uncertainty regions
    beta = 2.0
    ucb = mu + beta * sigma_safe

    # Cost-aware: divide by cost (lower cost is better)
    # Use log cost to soften the penalty for expensive points
    log_cost = np.log(cost + 1.0)
    scores = ucb / log_cost

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
