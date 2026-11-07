"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time to ensure we return before 330s
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB)
    with a dynamic exploration bonus, normalized by evaluation cost.
    
    The score is -(mu - kappa * sigma) / cost, where kappa is set to 1.0.
    This explicitly trades off exploitation of low means against exploration
    of high uncertainty while strictly penalizing expensive evaluations.
    """
    # Check deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:  # Safety margin of 5 seconds before 330s limit
        # Return scores that prioritize the current best point
        # This is a fallback to avoid timeout
        best_idx = np.argmin(mu)
        scores = np.zeros_like(mu)
        scores[best_idx] = 1.0
        return scores
    
    # Ensure sigma is safe for operations
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # LCB for minimization: lower is better
    # We want to minimize, so we prefer low mu and high sigma (exploration)
    # LCB = mu - kappa * sigma
    # Score = -LCB / cost = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    kappa = 1.0
    
    # Calculate LCB-based score
    lcb = mu - kappa * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
