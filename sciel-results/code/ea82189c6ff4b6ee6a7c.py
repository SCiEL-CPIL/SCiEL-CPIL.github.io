"""Rewritten program for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) for minimization.
    score = mu - kappa * sigma
    Higher score means more promising (lower expected value with exploration bonus).
    """
    # Check deadline - if we're close to timeout, return a simple deterministic score
    if time.time() > _DEADLINE:
        # Return scores based on mu only (pure exploitation) to finish quickly
        return -mu.astype(np.float64)
    
    kappa = 1.5
    # LCB for minimization: we want to minimize, so lower (mu - kappa*sigma) is better
    # But we need higher score = more promising, so we negate or use (mu - kappa*sigma) directly
    # Actually for minimization with "higher score = evaluate next", we want:
    # score = -(mu - kappa*sigma) = -mu + kappa*sigma
    # OR equivalently, score = kappa*sigma - mu
    scores = kappa * sigma - mu
    return scores
# EVOLVE-BLOCK-END
