"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining:
    - Expected Improvement (EI) for exploitation
    - Probability of Improvement (PI) for moderate exploration
    - Upper Confidence Bound (UCB) for exploration
    
    The cost penalty is softened (cost^0.5) to allow better exploration in early stages.
    An adaptive exploration weight increases early on and decreases as budget is used.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (scaled)
    pi = norm.cdf(z)
    
    # Upper Confidence Bound (exploration)
    # Use a moderate beta for UCB
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # Adaptive exploration weight based on cost usage
    # Early on (low cost), explore more; later, exploit more
    mean_cost = np.mean(cost)
    # Normalize cost to [0, 1] roughly
    cost_norm = np.clip((cost - 1.0) / 2.0, 0.0, 1.0)
    # Exploration weight: higher early, lower late
    # Use a smooth decay
    exploration_weight = 1.0 - 0.7 * cost_norm
    
    # Blend the acquisition functions
    # Weight EI heavily, but add PI and UCB components
    score_ei = ei / np.sqrt(cost_safe)
    score_pi = pi / np.sqrt(cost_safe)
    score_ucb = (f_best - ucb) / np.sqrt(cost_safe)
    
    # Combine: EI is primary, PI and UCB provide exploration
    combined = (
        0.6 * score_ei +
        0.25 * score_pi * exploration_weight +
        0.15 * score_ucb * exploration_weight
    )
    
    # Ensure finite output
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
