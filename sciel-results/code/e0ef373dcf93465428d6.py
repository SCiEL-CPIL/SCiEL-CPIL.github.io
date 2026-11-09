"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit in seconds
_SAFETY_MARGIN = 15.0  # seconds before deadline to start returning


def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function returning 2000 finite scores.
    
    Uses cost-normalized Probability of Improvement (POI).
    POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    Dividing by cost directly favors cheaper evaluations.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    # Cost ranges from 1 to 2, so this gives a meaningful relative penalty
    acq = poi / cost
    
    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    # Ensure non-negative
    acq = np.maximum(acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
