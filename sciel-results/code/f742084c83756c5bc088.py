import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Uncertainty-weighted exploration with cost awareness
    - Adaptive balance based on convergence state
    """
    t_start = time.time()
    deadline = 325.0 - 15.0  # 15 second safety margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma and cost to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute cost-normalized EI
    ei_cost = ei / cost_safe

    # Compute uncertainty (sigma) normalized by cost
    uncertainty_cost = sigma_safe / cost_safe

    # Adaptive weights based on convergence
    ei_max = np.max(ei_cost) if len(ei_cost) > 0 else 0.0
    ei_mean = np.mean(ei_cost) if len(ei_cost) > 0 else 0.0
    
    # When EI is concentrated (high max/mean ratio), we're in exploitation mode
    # When EI is spread out, we need more exploration
    if ei_max > 1e-10:
        concentration = ei_mean / ei_max
        # High concentration -> more exploitation, low concentration -> more exploration
        exploit_weight = 0.6 + 0.4 * np.clip(concentration, 0, 1)
        explore_weight = 1.0 - exploit_weight
    else:
        exploit_weight = 0.3
        explore_weight = 0.7

    # Combine EI and uncertainty with adaptive weights
    # Use sqrt(uncertainty) to avoid over-exploration
    exploration = explore_weight * np.sqrt(uncertainty_cost)
    exploitation = exploit_weight * ei_cost

    total_score = exploitation + exploration

    # Add a small bonus for low-cost points to encourage efficient exploration
    # This helps in the early stages when we're still mapping the landscape
    cost_bonus = 0.05 / (1.0 + cost_safe)
    total_score = total_score + cost_bonus

    # Ensure finite output
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return a safe default to avoid timeout
        return np.zeros_like(total_score)

    return total_score
