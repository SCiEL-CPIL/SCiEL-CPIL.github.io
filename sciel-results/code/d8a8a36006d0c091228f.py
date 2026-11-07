"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Cost-Adjusted Expected Improvement (EI) strategy. EI computes the expected
improvement over the best observed value under the GP posterior, providing a
natural balance between exploitation and exploration. The cost adjustment
divides by evaluation cost to favor cheaper candidates.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal
    
    improvement = f_best - mu  # Positive when mu < f_best (potential improvement)
    z = improvement / sigma
    
    # Compute EI using standard normal CDF and PDF
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # EI is non-negative; zero when no expected improvement
    ei = np.maximum(ei, 0.0)
    
    # Cost adjustment: divide by cost to favor cheaper evaluations
    score = ei / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
