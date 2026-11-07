"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 325s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a weighted-sum Pareto front approximation
    to explicitly trade off improvement magnitude against computational expense.
    
    mu, sigma, cost: shape (2000,). Higher score -> evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)

    n = mu.size

    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)

    # Standard Expected Improvement in objective units
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Multi-objective approach:
    # Objective 1: Maximize EI (improvement)
    # Objective 2: Minimize cost (expense)
    #
    # We use a weighted-sum Pareto front approximation.
    # First, normalize both objectives to [0, 1] range.
    
    # Normalize EI: higher is better
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range

    # Normalize cost: lower is better, so we invert
    cost_max = np.max(cost)
    cost_min = np.min(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost)
    else:
        cost_norm = (cost_max - cost) / cost_range  # Invert: lower cost -> higher norm

    # Weighted sum with adaptive weights:
    # Early on (high uncertainty), weight exploration (cost) more heavily
    # Late on (low uncertainty), weight exploitation (EI) more heavily
    
    # Use median sigma as a proxy for overall uncertainty
    sigma_median = np.median(sigma)
    
    # When sigma is large, we want to explore more -> weight cost more
    # When sigma is small, we want to exploit -> weight EI more
    # Use a smooth transition based on sigma_median relative to f_best scale
    
    f_scale = abs(f_best) + 1e-9
    
    # Weight for EI (exploitation)
    # High when sigma is small (confident), low when sigma is large (uncertain)
    w_ei = 1.0 / (1.0 + sigma_median / f_scale)
    
    # Weight for cost (exploration)
    w_cost = 1.0 - w_ei
    
    # Ensure weights are in a reasonable range
    w_ei = np.clip(w_ei, 0.3, 0.7)
    w_cost = 1.0 - w_ei

    # Pareto score: weighted sum of normalized objectives
    # Higher EI norm and higher cost norm (lower actual cost) -> higher score
    score = w_ei * ei_norm + w_cost * cost_norm

    # Add a small bonus for points with very high EI relative to their cost
    # This helps identify truly promising cheap points
    ei_per_cost = ei / cost
    ei_per_cost_max = np.max(ei_per_cost)
    if ei_per_cost_max > 1e-12:
        bonus = 0.1 * (ei_per_cost / ei_per_cost_max)
    else:
        bonus = np.zeros_like(ei_per_cost)
    
    score = score + bonus

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
