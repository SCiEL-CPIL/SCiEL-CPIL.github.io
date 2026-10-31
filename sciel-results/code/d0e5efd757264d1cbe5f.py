"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 10.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a Weibull Survival Function based heuristic.
    Prefers bins with smaller remaining capacity (Best-Fit variant)
    but with a non-linear penalty that aligns with Weibull item distributions.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        remaining = bins - item
        scores = np.full_like(remaining, -np.inf)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    scores = np.full_like(remaining, -np.inf)
    valid = remaining >= 0
    
    if np.any(valid):
        # Parameters for Weibull Survival Function S(r) = exp(-(r/lambda)^k)
        # lambda=10, k=1 corresponds to Exponential distribution with mean 10.
        # This penalizes remainders non-linearly, preferring very small remainders.
        lam = 10.0
        k = 1.0
        scores[valid] = np.exp(-(remaining[valid] / lam) ** k)
        
    return scores
# EVOLVE-BLOCK-END
