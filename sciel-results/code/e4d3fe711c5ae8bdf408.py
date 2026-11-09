"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Cost-Adjusted Lower Confidence Bound (LCB)
    # The exploration term is scaled by 1/sqrt(cost), treating expensive evaluations
    # as higher-variance observations in a Bayesian decision framework.
    # This is structurally different from simple division by cost.
    
    # Base LCB: lower confidence bound for minimization
    # We want to minimize, so we use mu - beta * sigma
    # But we need to handle the cost adjustment
    
    # Adaptive beta: increase exploration early, decrease as f_best improves
    # f_best starts high (rastrigin values can be ~20-40) and decreases toward 0
    f_best_abs = abs(f_best)
    # Scale beta based on how good our best solution is
    # When f_best is large, we're early and should explore more
    # When f_best is small, we're close to optimum and should exploit
    beta_base = 2.0
    
    # Cost adjustment: 1/sqrt(cost) penalizes expensive regions more strongly
    # in the exploration term, making the model treat high-cost areas as
    # having higher effective variance
    cost_safe = np.maximum(cost, 1e-6)
    cost_exploration_scale = 1.0 / np.sqrt(cost_safe)
    
    # The adjusted LCB: mu - beta * sigma * cost_scale
    # This means in high-cost regions, the exploration bonus is reduced,
    # making those regions less attractive unless mu is very low
    lcb = mu - beta_base * sigma * cost_exploration_scale
    
    # Also add a small exploitation component based on expected improvement
    # to ensure we still target promising regions
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] for combination
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei) * 0.5
    
    # Normalize LCB to [0, 1]
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = np.ones_like(lcb) * 0.5
    
    # For minimization, lower LCB is better, so we invert for scoring
    # Higher score = more likely to be evaluated next
    # We want to maximize the score, so we use 1 - lcb_norm (since lower LCB is better)
    # But we also want to combine with EI which is already "higher is better"
    
    # Weighted combination: balance exploration (LCB) and exploitation (EI)
    # Early on, explore more; later, exploit more
    exploit_weight = min(1.0, f_best_abs / 15.0)
    explore_weight = 1.0 - exploit_weight
    
    scores = explore_weight * (1.0 - lcb_norm) + exploit_weight * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
