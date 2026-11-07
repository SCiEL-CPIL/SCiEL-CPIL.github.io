"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    with adaptive exploration based on cost and uncertainty.
    
    UCB is often more robust than EI for minimization tasks, especially
    when the function has multiple local minima or when we want to balance
    exploration and exploitation more explicitly.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # UCB for minimization: we want low mu + beta * sigma
    # But we also need to be cost-aware
    # 
    # A common approach is to use a weighted combination:
    # score = -mu / cost + beta * sigma / cost
    # 
    # However, a more principled approach for cost-aware UCB is to consider
    # the expected improvement per unit cost. Let's use a variant that
    # balances the expected value with uncertainty, normalized by cost.
    
    # For minimization, we want to find points with low mu and high sigma
    # (uncertainty means potential for better values).
    
    # Compute a normalized uncertainty term
    # We scale beta based on the current best to make it adaptive
    # When f_best is close to the minimum, we want less exploration
    # When f_best is far, we want more exploration
    
    # Use a simple adaptive beta: start with moderate exploration, 
    # decrease as we get closer to the minimum
    # Since we don't know the true minimum, we use f_best as a proxy
    
    # A robust choice: beta = 1.0 * (1 + 0.1 * np.log1p(f_best))
    # But this might be too complex. Let's use a simpler approach.
    
    # Standard UCB for minimization:
    # ucb_value = mu - beta * sigma  (lower is better)
    # But we want higher score to be better, so:
    # score = -(mu - beta * sigma) / cost = (-mu + beta * sigma) / cost
    
    # Let's use beta = 2.0 as a reasonable default for 4D problems
    beta = 2.0
    
    # Compute UCB score
    ucb_score = (-mu + beta * sigma_safe) / cost_safe
    
    # Add a small bonus for points with very high uncertainty to ensure exploration
    # This helps when the GP is still learning the landscape
    uncertainty_bonus = 0.05 * sigma_safe / cost_safe
    
    final_score = ucb_score + uncertainty_bonus
    
    # Ensure all scores are finite
    final_score = np.where(np.isfinite(final_score), final_score, -np.inf)
    
    return final_score
# EVOLVE-BLOCK-END
