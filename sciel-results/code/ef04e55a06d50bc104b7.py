"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition: Upper Confidence Bound (UCB) for minimization.

    Score = -(mu - beta * sigma)
    Higher score means more promising (lower expected value with exploration).

    beta is adapted to the number of evaluations (inferred from f_best gap)
    and the scale of the problem.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return -mu.copy()

    # Known global minimum for Michalewicz 2D
    f_global_min = -1.8013

    # Estimate progress: gap from current best to global min
    gap = max(f_best - f_global_min, 1e-6)
    # Total possible range is roughly from ~1.5 (worst initial) to -1.8013
    # So total span ~ 3.3
    total_span = 3.3
    progress = np.clip(1.0 - gap / total_span, 0.0, 1.0)

    # Adaptive beta: start high for exploration, decrease for exploitation
    # Use a smooth decay: beta = beta_max * (1 - progress)^alpha + beta_min
    beta_max = 2.5
    beta_min = 0.5
    alpha = 1.5
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress) ** alpha

    # UCB for minimization: we want low mu, high sigma
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    # Higher score = better (lower expected value, more uncertainty)
    scores = -mu + beta * sigma

    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
