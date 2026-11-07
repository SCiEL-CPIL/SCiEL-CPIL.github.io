"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 15.0  # seconds to leave before the limit


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (-sigma_safe) / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement (EI) for minimization
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    ei_cost = ei / cost_safe

    # Compute UCB for minimization (Lower Confidence Bound)
    # Use a fixed beta of 2.0 which showed better performance
    beta = 2.0
    ucb_min = mu - beta * sigma_safe
    ucb_score = -ucb_min / cost_safe

    # Use rank-based normalization for robustness
    # This creates a Pareto-like front where points need to be good on both dimensions
    ei_ranks = np.argsort(np.argsort(ei_cost))
    ucb_ranks = np.argsort(np.argsort(ucb_score))
    
    # Normalize ranks to [0, 1]
    n = len(ei_cost)
    ei_norm = ei_ranks / (n - 1)
    ucb_norm = ucb_ranks / (n - 1)

    # Blend: 70% cost-aware EI, 30% cost-aware UCB for exploration
    weight_ei = 0.7
    weight_ucb = 0.3

    scores = weight_ei * ei_norm + weight_ucb * ucb_norm

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
