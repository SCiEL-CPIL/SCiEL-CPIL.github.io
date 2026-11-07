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
        # Return a simple fallback: negative sigma (uncertainty) divided by cost
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

    # Compute Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe

    # Numerically stable EI computation
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    ei_cost = ei / cost_safe

    # UCB component for exploration in minimization:
    # We want to explore where mu is low AND sigma is high
    # ucb_min = mu - beta * sigma (lower is better)
    # Score: -ucb_min / cost (higher score = better)
    
    # Adaptive beta: use larger beta early (high uncertainty), smaller later
    # Estimate progress by how far f_best is from the known minimum
    # Hartmann 6D minimum is approximately -3.32237
    # Early: f_best is high (less negative), later: f_best approaches -3.32237
    # We can use the spread of mu as a proxy for exploration progress
    mu_spread = np.std(mu)
    # If mu_spread is large, we're still exploring, so use higher beta
    # If mu_spread is small, we're converging, so use lower beta
    # Scale beta between 1.0 and 3.0 based on mu_spread
    beta = 1.0 + 2.0 * np.clip(mu_spread / (np.abs(f_best) + 1e-9), 0.0, 1.0)
    
    ucb_min = mu - beta * sigma_safe
    ucb_score = -ucb_min / cost_safe

    # Normalize both components to [0, 1] range for blending
    ei_min = np.min(ei_cost)
    ei_max = np.max(ei_cost)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei_cost - ei_min) / ei_range

    ucb_min_val = np.min(ucb_score)
    ucb_max_val = np.max(ucb_score)
    ucb_range = ucb_max_val - ucb_min_val + 1e-9
    ucb_norm = (ucb_score - ucb_min_val) / ucb_range

    # Blend: 70% cost-aware EI, 30% cost-aware UCB for exploration
    weight_ei = 0.7
    weight_ucb = 0.3

    scores = weight_ei * ei_norm + weight_ucb * ucb_norm

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
