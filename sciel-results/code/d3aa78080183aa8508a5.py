import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    Uses Quantile-Based Cost-Effective Acquisition:
    Computes the 10th percentile of the predicted objective distribution
    (q_0.1 = mu - 1.28*sigma) and maximizes -q_0.1/cost.
    This targets high-confidence minimization of cost-adjusted performance.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma and cost
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # Compute 10th percentile of the predicted objective
    # q_0.1 = mu - 1.28 * sigma (since 1.28 is the z-score for 0.90 quantile)
    q_01 = mu - 1.28 * sigma
    
    # Score: negative quantile divided by cost
    # Higher score means lower expected cost-adjusted objective
    scores = -q_01 / cost

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
