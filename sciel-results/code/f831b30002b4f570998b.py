"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective Expected Improvement (CEI).
    Returns 2000 finite scores; higher = evaluate next.
    
    Strategy:
    - Compute Expected Improvement (EI) for minimization.
    - Divide by the evaluation cost to get CEI.
    - This balances the expected reduction in objective value against the cost of evaluation.
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: return uniform scores or favor cheap points
        return 1.0 / np.maximum(cost, 1e-9)

    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Expected Improvement for minimization
    # d = f_best - mu
    # If d > 0, there is potential for improvement.
    d = f_best - mu
    Z = d / s

    # EI = d * Phi(Z) + s * phi(Z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    EI = d * norm.cdf(Z) + s * norm.pdf(Z)

    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)

    # Cost-Effective EI
    CEI = EI / c

    # Ensure finite
    CEI = np.where(np.isfinite(CEI), CEI, 0.0)
    
    return CEI
# EVOLVE-BLOCK-END
