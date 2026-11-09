"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _safe_divide(a, b, eps=1e-9):
    """Safe element-wise division with epsilon guard."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Lower Confidence Bound (LCB) for minimization, which explicitly
    rewards high uncertainty regions. This is better suited for multimodal
    landscapes like Griewank than EI, which focuses on expected gain magnitude.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: mu - beta * sigma
    # Beta controls the exploration-exploitation tradeoff
    # A higher beta encourages more exploration
    beta = 2.0
    
    lcb = mu - beta * sigma
    
    # Cost-aware: divide by square root of cost to reduce bias towards cheap corners
    # Cost is in [1, 2]. sqrt(Cost) is in [1, 1.414].
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = np.sqrt(cost_safe)
    
    # For minimization, we want to maximize the acquisition score
    # LCB gives lower values for better (lower) predicted f
    # So we negate it to make higher scores better
    score = -lcb / cost_factor
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
