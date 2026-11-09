import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) style acquisition for Goldstein-Price 2D.
    
    The score is the probability that the true function value at candidate x
    is better (lower) than a dynamic threshold slightly below f_best.
    This naturally balances exploration (high sigma regions) and exploitation
    (low mu regions) and is well-suited for minimization problems.
    
    The threshold is set adaptively: as f_best improves, the threshold moves
    lower, encouraging the search to find even better points.
    """
    start_time = time.time()
    deadline = start_time + 2.0
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety: ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Safety: ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Dynamic threshold: set below f_best to encourage finding new minima
    # The offset is proportional to the spread of sigma to adapt to uncertainty
    sigma_mean = float(np.mean(sigma))
    # Start with a modest offset, reduce as we have more evaluations (f_best improves)
    # Use a fraction of the current sigma spread to keep threshold meaningful
    threshold = f_best - 0.5 * sigma_mean
    
    # Compute PLB score: P(f(x) < threshold) = Phi((threshold - mu) / sigma)
    # where Phi is the standard normal CDF
    z = (threshold - mu) / sigma
    scores = norm.cdf(z)
    
    # Ensure scores are finite and in [0, 1]
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, 0.0, 1.0)
    
    # If deadline is close, return immediately
    if time.time() > deadline:
        return scores
    
    return scores
