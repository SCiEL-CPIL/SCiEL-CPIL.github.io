"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Leave 20s for cleanup/other overhead


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    
    Strategy: Cost-aware Expected Improvement (EI) with principled cost penalty.
    Uses EI as the base acquisition, then applies a global cost penalty based on
    the cost distribution to prefer cost-effective evaluations.
    """
    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score: just exploit (negative mu for minimization)
        return -mu.copy()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, EI is positive when mu < f_best
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best, but we clip to 0 for acquisition
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware penalty: use global cost z-score
    # Penalize points with cost above the mean
    mean_cost = np.mean(cost)
    std_cost = np.std(cost)
    
    if std_cost > 1e-12:
        cost_z = (cost - mean_cost) / std_cost
        # Penalty: exponential decay for high-cost points
        # This is a soft penalty that's 1 at mean cost and decays for expensive points
        penalty = np.exp(-0.5 * np.maximum(cost_z, 0.0) ** 2)
        score = ei * penalty
    else:
        score = ei.copy()
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
