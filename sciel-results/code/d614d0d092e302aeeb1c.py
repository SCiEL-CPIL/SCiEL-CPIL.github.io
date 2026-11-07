"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s. We track elapsed time and
# ensure we return well before that.
_DEADLINE_SECONDS = 325.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function for minimization.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    For minimization, we want points where the probability of beating the
    current best is high. This gives a strong exploitation bias while
    still accounting for uncertainty.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior standard deviation in objective units
        f_best: float - lowest observed value so far
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    # Check deadline
    if _elapsed() > _DEADLINE_SECONDS:
        # Return a safe fallback: just use negative mu (exploitation only)
        return -mu.copy()
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-12)
    
    # PI for minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Where Phi is the standard normal CDF
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with the CDF
    z = np.clip(z, -10, 10)
    
    # Compute PI scores
    scores = norm.cdf(z)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
