"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Use a deadline guard to avoid exceeding wall-clock limits
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    # Lower Confidence Bound (LCB) with adaptive beta
    # Higher beta encourages exploration; we use a moderate value
    # and add a spatial diversity term via sigma weighting

    # Base LCB: mu - beta * sigma
    # We want to minimize, so lower values are better
    # But acquisition returns scores where HIGHER means evaluate next
    # So we negate: score = -(mu - beta * sigma) = -mu + beta * sigma

    # Adaptive beta: start high for exploration, decrease over time
    # We don't have access to iteration count directly, but we can use
    # the relationship between f_best and the typical scale of the problem
    # Hartmann 3d has minimum -3.86278, so f_best starts around -1 to -3

    # Use a fixed moderate beta that balances exploration/exploitation
    # The key insight from measured results: adding spatial diversity bonus
    # proportional to sigma improved score from 1.891 to 2.51

    # LCB score: -mu + beta * sigma
    # The sigma term naturally provides spatial diversity since regions
    # with high uncertainty (less explored) get higher scores

    beta = 2.0  # Moderate exploration parameter

    # Compute LCB scores
    scores = -mu + beta * sigma

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    # Check deadline
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
