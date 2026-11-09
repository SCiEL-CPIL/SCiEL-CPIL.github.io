"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return indices of lowest mu to at least try exploitation
        # This prevents timeout crashes
        return -mu
    
    # Lower Confidence Bound (LCB) for minimization
    # Score = mu - beta * sigma
    # Higher score is preferred.
    # We want low mu (exploitation) and high sigma (exploration).
    
    # Clip sigma to avoid numerical issues, though LCB is more stable than EI
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Beta parameter controls exploration/exploitation trade-off
    # A value of 2.0 is a reasonable default for 2D smooth functions.
    # We could make this adaptive, but a fixed structural change is requested.
    beta = 2.0
    
    lcb = mu - beta * sigma_safe
    
    # Ensure scores are finite
    lcb = np.where(np.isfinite(lcb), lcb, -np.inf)
    
    return lcb
# EVOLVE-BLOCK-END
