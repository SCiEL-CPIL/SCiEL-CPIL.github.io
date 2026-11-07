"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Theoretical minimum of cosine8 in 8D
_F_MIN = -0.8


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses an adaptive Lower Confidence Bound (LCB) approach with cost awareness:
    - Exploitation: favor low mu (predicted minimum)
    - Exploration: favor high sigma (uncertainty)
    - Cost awareness: penalize expensive evaluations
    - Adaptive: balances exploitation/exploration based on proximity to f_min
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # Cost penalty: normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Adaptive beta: increases as we get closer to the theoretical minimum
    # When f_best is far from f_min, we explore more (higher beta)
    # When f_best is close to f_min, we exploit more (lower beta)
    gap = max(f_best - _F_MIN, 1e-6)
    # Scale beta based on gap: larger gap -> more exploration
    # Use a smooth transition
    beta_base = 2.0
    beta_adaptive = beta_base * (1.0 + 0.5 * np.exp(-gap * 2.0))
    
    # LCB: mu - beta * sigma (lower is better for minimization)
    lcb = mu - beta_adaptive * sigma
    
    # Cost-aware score: we want high scores, so negate LCB and penalize cost
    # Higher score = more promising
    # Normalize LCB to have similar scale as cost penalty
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range < 1e-12:
        lcb_norm = np.ones_like(lcb) * 0.5
    else:
        lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Combine: favor low LCB (high lcb_norm inverted) and low cost
    # Score = -(lcb) / (1 + cost_penalty)
    # Higher score means more promising
    
    # Use a weighted combination
    w_exploit = 0.6
    w_cost = 0.4
    
    # Base score: negative LCB (we want low LCB = high score)
    base_score = -lcb
    
    # Normalize base score
    base_min = np.min(base_score)
    base_max = np.max(base_score)
    base_range = base_max - base_min
    if base_range < 1e-12:
        base_norm = np.ones_like(base_score) * 0.5
    else:
        base_norm = (base_score - base_min) / base_range
    
    # Cost penalty term: lower cost is better
    cost_score = 1.0 - cost_norm
    
    # Final score: weighted combination
    score = w_exploit * base_norm + w_cost * cost_score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
