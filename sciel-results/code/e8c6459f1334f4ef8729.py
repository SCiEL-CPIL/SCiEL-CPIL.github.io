# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a Probability of Improvement (PI) acquisition function.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)

    # Ensure all scores are finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        return -mu
    
    return pi
# EVOLVE-BLOCK-END
