import numpy as np
import time
from scipy.stats import norm

# Global start time for the deadline guard (set on first call).
_START_TIME = None
# Hard wall-clock limit (seconds) for the whole evaluation.
_HARD_LIMIT = 330.0
# Safety margin so we return well before the limit.
_MARGIN = 5.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    PI is a greedy acquisition function that selects the point with the
    highest probability of improving upon the current best observed value.
    
    PI(x) = Phi((f_best - mu(x)) / sigma(x))
    
    This is a structural change from the previous EI + UCB hybrid,
    shifting towards a more exploitative strategy with adaptive exploration.
    """
    elapsed = _elapsed()
    if elapsed > _HARD_LIMIT - _MARGIN:
        # Deadline exceeded, return a simple score
        # Fallback to a greedy score based on mu
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle edge cases where sigma is very small
    sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
    
    # Compute Probability of Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # PI = Phi(z)
    pi = norm.cdf(z_clipped)
    
    # Ensure finite values
    scores = np.where(np.isfinite(pi), pi, 0.0)
    
    return scores
