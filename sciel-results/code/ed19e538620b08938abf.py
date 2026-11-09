import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Probabilistic Lower Bound (PLB): maximize P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    which is equivalent to maximizing (f_best - mu) / sigma.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive with a reasonable floor
    sigma = np.maximum(sigma, 1e-6)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0
    
    # PLB score: (f_best - mu) / sigma
    # This is the z-score such that P(f(x) < f_best) = Phi(z)
    # Maximizing this maximizes the probability of improvement.
    
    # Compute the raw z-scores
    scores = (f_best - mu) / sigma
    
    # Handle cases where mu is already below f_best (shouldn't happen often, but be safe)
    # If mu < f_best, the score is positive and large, which is correct.
    # If mu >> f_best, the score is negative, which is correct.
    
    # For very small sigma, the score can be extremely large or small.
    # Clip to prevent numerical issues.
    scores = np.clip(scores, -1e6, 1e6)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
