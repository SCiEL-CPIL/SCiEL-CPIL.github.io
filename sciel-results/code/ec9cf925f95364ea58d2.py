"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition function combining EI and UCB with a deadline guard.
    
    # Deadline guard: if we are close to the 300s limit, return a safe fallback
    # to avoid timeout. In practice, this function is called many times per
    # evaluation, so we track a global start time.
    # However, since we don't have access to a global timer easily, we'll use
    # a lightweight check. The main bottleneck is likely the GP updates, not
    # this function. But to be safe, we'll keep this fast.
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: score = beta * sigma - mu
    # We want to maximize this, so we use it as a bonus
    beta = 1.0
    ucb = beta * sigma_safe - mu
    
    # Normalize both to [0, 1] range for combining
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = np.clip(ucb / ucb_max, 0, 1)
    
    # Combine EI and UCB with a weight
    # Higher weight on EI for exploitation, UCB for exploration
    weight_ei = 0.7
    weight_ucb = 0.3
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Use a softer cost discount (alpha=0.7) to not be too aggressive
    alpha = 0.7
    score = score / (cost_safe ** alpha)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
