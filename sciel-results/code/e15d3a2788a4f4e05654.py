"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds of safety margin
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition with dynamically annealed exploration parameter xi.
    xi starts at 0.1 and decays linearly to 0.0 over 25 optimization steps.
    Higher score = higher expected improvement = more likely to be evaluated next.
    """
    global _call_count, _start_time
    
    # Check deadline
    if time.time() - _start_time > 330 - _DEADLINE_MARGIN:
        # Return a fallback score based on negative mean
        return -mu
    
    _call_count += 1
    
    # Dynamically anneal xi from 0.1 to 0.0 over 25 steps
    # The first 5 evaluations are initial points, so optimization steps start at call 6
    # But we'll just use _call_count directly, assuming it tracks total calls
    # Total optimization steps = 30 - 5 = 25
    if _call_count <= 5:
        # Initial points: use high exploration
        xi = 0.1
    else:
        step_in_opt = _call_count - 5
        if step_in_opt >= 25:
            xi = 0.0
        else:
            xi = 0.1 * (1.0 - step_in_opt / 25.0)
    
    # Handle non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    # But we need to be careful: EI is the expected improvement over f_best
    # For minimization, improvement = max(0, f_best - f(x))
    # EI = E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Add exploration parameter xi: EI(xi) = EI - xi * sigma
    # This encourages exploration by subtracting a term proportional to uncertainty
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_with_xi = ei - xi * sigma_safe
    
    return ei_with_xi
# EVOLVE-BLOCK-END
