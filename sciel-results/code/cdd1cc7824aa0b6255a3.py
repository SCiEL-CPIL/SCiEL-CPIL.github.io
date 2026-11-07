"""Rewritten acquisition: cost-aware UCB with adaptive exploration, structural change from EI."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound acquisition for minimization.
    Uses a different structural approach than EI: UCB balances exploitation
    (low mu) and exploration (high sigma) with cost normalization.
    """
    start_time = time.time()
    deadline = 300.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safe sigma: avoid division by zero, clip to positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Safe cost: avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)

    # Compute UCB for minimization: we want low mu + high uncertainty
    # UCB_min = mu - beta * sigma  (lower is better for minimization)
    # But we need a score where HIGHER = evaluate next
    
    # Adaptive beta: starts high for exploration, decays as we find better solutions
    # Use the spread of mu to adapt beta
    mu_range = np.ptp(mu)
    if mu_range < 1e-12:
        mu_range = 1.0
    
    # Beta parameter: balance exploitation vs exploration
    # Scale by typical uncertainty relative to objective range
    sigma_mean = np.mean(sigma_safe)
    beta = 1.5 + 0.5 * (sigma_mean / (mu_range + 1e-12))
    
    # UCB score for minimization: higher score = more promising
    # We negate the UCB_min so that higher values are better
    # UCB_min = mu - beta * sigma, so score = -(mu - beta*sigma) = -mu + beta*sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    # But also add a small exploration bonus for high sigma to avoid getting stuck
    cost_score = ucb_score / cost_safe
    
    # Add a small exploration component: prefer points with high sigma
    # This helps escape local optima in the cosine8 landscape
    exploration_bonus = 0.1 * (sigma_safe / (np.max(sigma_safe) + 1e-12))
    
    score = cost_score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > (deadline - 5.0):
        return score
    
    return score
# EVOLVE-BLOCK-END
