import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    Uses a pure exploitation strategy that prioritizes lowest predicted values,
    allowing exploration of high-cost regions where shifted minima may reside.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-8)

    # Hybrid strategy: Exploitation with slight exploration
    # Use UCB with a small beta to avoid getting stuck in local GP minima
    beta = 0.1
    scores = -mu + beta * sigma

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
