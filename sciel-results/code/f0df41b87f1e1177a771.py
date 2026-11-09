# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 10.0  # seconds before the hard limit (300s)

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290.0  # 300s limit, 10s safety margin
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    Strategy:
    1. Calculate Expected Improvement (EI) for minimization.
    2. Normalize by cost to prefer cheaper evaluations that still offer improvement.
    3. Ensure numerical stability and finite outputs.
    """
    # Check time remaining to avoid timeout
    if _time_remaining() < 0.1:
        # Return a safe, finite score if time is almost up
        return np.zeros(len(mu))
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use scipy.stats.norm for accurate CDF and PDF
    # To improve speed and stability, we can use vectorized operations
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # EI can be negative if mu > f_best, but we want to select the best candidate.
    # In practice, EI is 0 when mu >= f_best (since we only care about improvement).
    # However, the formula naturally yields small positive values for mu slightly above f_best due to numerical noise or the tail.
    # We should clip EI to be non-negative to avoid rewarding bad points.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is given as (1 + mean(x scaled to [0, 1]))
    # We want to maximize EI / cost
    score = ei / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
