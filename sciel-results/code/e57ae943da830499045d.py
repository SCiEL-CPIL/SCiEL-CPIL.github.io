"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard to ensure we finish well before the 300s/330s limit
_start_time = time.time()
_DEADLINE = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    """
    Computes the Lower Confidence Bound (LCB) acquisition score.
    For minimization, we want to find points where the lower bound of the
    function value is as low as possible.
    LCB = mu - kappa * sigma
    """
    # Check time budget
    if time.time() > _DEADLINE:
        # If we are running out of time, return a safe, finite score
        # to avoid crashing the evaluator, though ideally we should be fast.
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid numerical issues, though GP should provide > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive or Fixed kappa?
    # In low-budget (30 evals) 2D problems, a fixed kappa around 1.0-2.0 is often robust.
    # A value of 1.0 is a good standard starting point for LCB.
    # We can also consider a slight adaptive component, but fixed is simpler and stable.
    kappa = 1.0
    
    # Calculate LCB score
    # Higher score means more "promising" for minimization (i.e., lower expected value)
    scores = mu - kappa * sigma_safe
    
    # Ensure finiteness
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
