"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function.
    
    Uses a UCB/LCB strategy which balances exploration and exploitation
    by adding a multiple of the standard deviation to the mean.
    Since we are minimizing, we use the Lower Confidence Bound:
    LCB = mu - beta * sigma
    We return the negative of this (or an equivalent score) so that
    higher scores correspond to lower predicted values (better).
    
    Score = -mu + beta * sigma
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value (not directly used in LCB, but kept for signature)
    
    Returns:
        array of shape (2000,), finite scores, higher means evaluate next
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple LCB-like score as a safe fallback
        return -mu - 0.1 * sigma

    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta parameter for exploration
    # A value around 2.5-3.0 is often effective for small budgets in 6D
    beta = 2.5
    
    # LCB score: we want to minimize mu - beta*sigma
    # So we return -mu + beta*sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
