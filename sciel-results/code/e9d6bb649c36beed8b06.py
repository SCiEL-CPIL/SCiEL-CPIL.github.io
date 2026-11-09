import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 285.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Deadline guard: If we are very close to the time limit,
    # return a score that strongly favors exploitation (low mu)
    # to avoid wasting time on exploration or complex calculations.
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to pure exploitation: lower mu is better
        # Invert mu so higher score = lower mu
        return -mu
    
    # Ensure sigma is safe for division and operations
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Use Upper Confidence Bound (UCB) for better exploration-exploitation balance
    # UCB = -mu + beta * sigma (minimization problem)
    # Beta is set to 1.5 to encourage more exploration than the default 1.0,
    # which is often beneficial in low-budget regimes for multimodal functions like Ackley.
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(ucb), ucb, 0.0)
    
    return score
