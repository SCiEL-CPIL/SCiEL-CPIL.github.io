"""Rewritten program using a pure LCB acquisition with a constant beta."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Pure LCB acquisition for minimization.
    Score = -mu + beta * sigma, with a constant beta.
    Higher score means more promising to evaluate next.
    """
    if not _elapsed_ok():
        return np.random.rand(len(mu))

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Constant beta for LCB
    # For minimization, LCB = mu - beta * sigma
    # We want to maximize -LCB = -mu + beta * sigma
    # A constant beta around 1.0-2.0 typically works well for BO
    beta = 1.5

    score = -mu + beta * sigma_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
