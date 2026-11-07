"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) with dampened cost penalty.
    
    Standard EI/cost can over-penalize points with higher costs (higher coordinates).
    Since the minimizer is uniformly distributed in the inner 80% of the domain,
    points in the center (medium cost) are highly probable locations for the minimum.
    Adding a small constant (0.1) to the cost denominator reduces the bias towards
    the cheapest points, allowing the search to explore the central regions more effectively.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Calculate z-score for EI
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI should be non-negative, but numerical errors might make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score with dampened penalty
    # cost is in [1, 2]. Adding 0.1 reduces the dynamic range of the penalty.
    cost_denom = cost + 0.1
    
    scores = ei / cost_denom
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
