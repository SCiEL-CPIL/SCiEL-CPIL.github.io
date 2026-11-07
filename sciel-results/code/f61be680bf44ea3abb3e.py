"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost
    as the primary signal, with a structured exploration bonus.
    
    The key insight: in cost-aware BO, we want to maximize the expected
    improvement per unit of cost spent. This is the theoretically optimal
    criterion when each evaluation has a different cost.
    """
    t0 = time.time()
    n = len(mu)
    
    # Input validation
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement in objective units
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: the primary signal
    # This directly measures expected improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Structured exploration: use a quantile-based approach
    # Points with high sigma relative to their EI get an exploration bonus
    # This avoids the pitfalls of fixed UCB weights
    
    # Compute a normalized uncertainty measure
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # Only explore in regions where we haven't seen much improvement yet
    # i.e., where mu is not already well below f_best
    improvement_ratio = np.maximum(f_best - mu, 0) / (np.max(f_best - mu) + 1e-10)
    
    # Exploration bonus: high when sigma is high AND improvement is uncertain
    # Use a smooth transition: explore more when the model is uncertain
    # and less when we're confident about the current best
    exploration_weight = 0.15 * sigma_norm * (1.0 - 0.5 * improvement_ratio)
    
    # Combine: EI/cost is dominant, exploration is secondary
    # Scale exploration to be comparable to ei_per_cost
    ei_per_cost_max = np.max(ei_per_cost) + 1e-10
    exploration_scaled = exploration_weight * ei_per_cost_max
    
    score = ei_per_cost + exploration_scaled
    
    # Deadline guard
    if time.time() - t0 > 280:
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
