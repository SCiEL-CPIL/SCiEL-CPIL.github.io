"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard: track when the process started.
# The evaluator kills at 330s wall-clock; we use a safety margin.
_PROCESS_START = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill to bail out safely
_DEADLINE = _PROCESS_START + 330.0 - _DEADLINE_MARGIN


def _time_remaining():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CA-EI) with power-law cost penalty.

    Structural change: Instead of a linear cost penalty or adaptive blending,
    we use EI / cost^gamma where gamma < 1 (sub-linear cost penalty).
    This directly optimizes improvement per unit cost while maintaining
    sensitivity to high-improvement points in low-cost regions.

    The sub-linear penalty (gamma=0.5) means that doubling the cost only
    increases the penalty by sqrt(2), so the acquisition still favors
    high-EI points even at moderate cost, but strongly penalizes very
    expensive evaluations.

    mu, sigma, cost: shape (2000,)
    f_best: scalar, lowest observed value so far.
    Returns: shape (2000,) finite scores, higher = more desirable.
    """
    # Deadline guard: if we're running out of time, fall back to a simple
    # cost-aware EI that is fast to compute.
    if _time_remaining() < 0.5:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / np.sqrt(cost_safe)
        return np.where(np.isfinite(score), score, 0.0)

    # Compute standard Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure non-negative EI (clip small negatives from numerical error)
    ei = np.maximum(ei, 0.0)

    # Cost-aware score with sub-linear (power-law) cost penalty
    # gamma = 0.5: sqrt(cost) penalty
    # This is a structural change from linear cost penalty to power-law
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = np.sqrt(cost_safe)  # cost^0.5
    score = ei / cost_penalty

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
