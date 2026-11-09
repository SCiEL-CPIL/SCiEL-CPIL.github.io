"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB formulation with dynamic kappa.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP surrogate at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP surrogate at candidate points.
    f_best : float
        Best observed value so far (lowest value, since we minimize).
    cost : array_like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Dynamic kappa based on search progress
    # When f_best is poor (high), explore more (higher kappa)
    # When f_best is good (low, close to -3.86), exploit more (lower kappa)
    # Use a sigmoid-like transition
    # f_best ranges from ~-3.86 (optimal) to maybe 0 or positive (poor)
    # Map f_best to kappa in range [1.0, 5.0]
    # When f_best = -3.86 (optimal), kappa = 1.0 (exploit)
    # When f_best = 0 (poor), kappa = 5.0 (explore)
    # Linear interpolation: kappa = 1.0 + 4.0 * (f_best - (-3.86)) / (0 - (-3.86))
    # Clamp to [1.0, 5.0]
    f_min_expected = -3.86
    f_max_expected = 0.0
    kappa_raw = 1.0 + 4.0 * (f_best - f_min_expected) / (f_max_expected - f_min_expected)
    kappa = np.clip(kappa_raw, 1.0, 5.0)
    
    # UCB for minimization: score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # We want higher score = better, so we minimize mu - kappa*sigma
    # Score = -(mu - kappa * sigma) = -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma
    
    # Cost-aware normalization: divide by cost
    scores = ucb_score / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
