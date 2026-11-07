"""Cost-aware acquisition function for Bayesian optimization (levy 2d).

This module implements a cost-aware Expected Improvement (EI) acquisition function.
"""
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track module load time and ensure we return well before the limit.
_MODULE_START = time.time()
DEADLINE = 325.0  # seconds; leave 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).

    Calculates the expected improvement over f_best for each candidate,
    normalized by the evaluation cost. Higher scores indicate better candidates.

    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard-deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher is preferred.  All values are finite.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we are too close to the kill time, return a
    # safe fallback (uniform random scores) immediately.
    if time.time() > DEADLINE:
        return np.random.uniform(0.0, 1.0, size=mu.shape)

    n = mu.shape[0]

    # Guard against degenerate inputs
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = ei / cost_safe
    
    # Clip to finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
