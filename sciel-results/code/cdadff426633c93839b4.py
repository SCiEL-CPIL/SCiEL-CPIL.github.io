"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time inside acquisition and bail out early if needed.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using GP-LCB.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value (unused in LCB, kept for interface)
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.zeros(len(mu))
    
    n = len(mu)
    eps = 1e-9
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # GP-LCB for minimization:
    # We want to minimize f(x).
    # The Lower Confidence Bound is L(x) = mu(x) - beta * sigma(x).
    # We select the point with the lowest L(x), i.e., min(mu - beta*sigma).
    # To fit the "maximize score" interface, we return score = -L(x) = -mu + beta * sigma.
    # Dividing by cost makes it cost-aware.
    
    beta = 2.0
    
    scores = (-mu + beta * sigma_safe) / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE:
        return np.zeros(n)
    
    return scores
# EVOLVE-BLOCK-END
