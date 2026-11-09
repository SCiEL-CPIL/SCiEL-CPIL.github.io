"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds total wall-clock
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition for minimization.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std in objective units
        f_best: float - best (lowest) observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: just use negative mu (exploit best known)
        return -mu.copy()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    
    # LCB for minimization: we want to minimize, so lower confidence bound is mu - beta * sigma
    # We return the negative so that higher score = more promising (lower predicted value)
    # 
    # Use a constant beta. For Michalewicz 2D with 25 evaluations total,
    # a moderate exploration constant works well.
    # beta=2.0 is a common choice that balances exploration/exploitation.
    # 
    # The LCB score: higher is better for selection
    # lcb = mu - beta * sigma (this is the lower bound on the function value)
    # We want to maximize the chance of finding low values, so we pick the point
    # with the lowest LCB (most optimistic prediction).
    # Return negative LCB so that higher return value = pick this point.
    
    beta = 2.0
    scores = -(mu - beta * sigma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
