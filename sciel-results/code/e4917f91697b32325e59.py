"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining normalized Expected Improvement (EI)
    and normalized Upper Confidence Bound (UCB) with adaptive weighting.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        # Simple greedy fallback: minimize mu per unit cost
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # 1. Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)

    # 2. Compute UCB score for minimization
    # We want to maximize (mu - kappa * sigma) -> Score is -(mu - kappa * sigma) = kappa * sigma - mu
    # Using a moderate kappa. Since we normalize later, the absolute scale matters less,
    # but kappa=2.0 is a standard starting point for exploration.
    kappa = 2.0
    ucb = kappa * sigma - mu

    # 3. Cost Awareness
    # We want to penalize high cost. Dividing by cost is a standard approach.
    ei_cost = ei / cost_safe
    ucb_cost = ucb / cost_safe
    
    # 4. Normalization
    # Normalize EI and UCB scores to [0, 1] range to make them comparable.
    # This prevents EI (which can be large) from dominating UCB (which can be small/negative).
    
    ei_min, ei_max = np.min(ei_cost), np.max(ei_cost)
    ucb_min, ucb_max = np.min(ucb_cost), np.max(ucb_cost)
    
    # Avoid division by zero
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei_cost)
    else:
        ei_norm = (ei_cost - ei_min) / ei_range
        
    if ucb_range < 1e-12:
        ucb_norm = np.zeros_like(ucb_cost)
    else:
        ucb_norm = (ucb_cost - ucb_min) / ucb_range

    # 5. Adaptive Weighting
    # Determine how much to weight exploration (UCB) vs exploitation (EI).
    # If the GP is highly uncertain (high sigma relative to max sigma), favor UCB.
    # If the GP is certain (low sigma), favor EI.
    
    sigma_max = np.max(sigma)
    if sigma_max < 1e-9:
        uncertainty_ratio = np.zeros_like(sigma)
    else:
        uncertainty_ratio = sigma / sigma_max
        
    # Weight for UCB (exploration) increases with uncertainty.
    # Range for ucb_weight: [0.3, 0.7]
    # This ensures we always have some exploration and some exploitation.
    ucb_weight = 0.3 + 0.4 * uncertainty_ratio
    ei_weight = 1.0 - ucb_weight

    # 6. Combine
    score = ei_weight * ei_norm + ucb_weight * ucb_norm

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
