"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamically tuned Gaussian Process Lower Confidence Bound (GP-LCB)
    normalized by cost. The exploration coefficient beta is adapted based on
    the current best value and remaining budget proxy.
    """
    # Deadline guard: if we're running too long, return a safe default
    # (This is a heuristic since we don't have explicit start time, 
    #  but the function is called repeatedly so we keep it fast)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # f_best is the lowest observed value (we want to minimize)
    # For LCB: score = mu - beta * sigma (lower is better for minimization)
    # We want higher scores to be selected, so we negate or use UCB-style
    # Actually for minimization: we want low mu and low sigma
    # LCB for minimization: mu - beta * sigma (we pick the smallest)
    # But our interface returns scores where HIGHER = evaluate next
    # So we should return: -(mu - beta * sigma) = -mu + beta * sigma
    # Or equivalently use an "Upper Confidence Bound" on the negative function
    
    # Let's think of it as maximizing: g(x) = -f(x)
    # Then mu_g = -mu, sigma_g = sigma
    # UCB for g: mu_g + beta * sigma_g = -mu + beta * sigma
    # This is what we return as the score (higher = better)
    
    # Dynamic beta: scale by 1/|f_best| to intensify exploration as f_best -> 0
    # Also consider cost normalization
    # f_best can be close to 0, so we need a floor
    f_best_abs = max(abs(f_best), 1e-6)
    
    # Base exploration coefficient
    # Start with a moderate value and scale
    beta_base = 1.0
    
    # Scale beta inversely with f_best: as f_best gets closer to 0, 
    # we want more exploration to escape local optima? Or less?
    # Actually, as we get closer to the optimum, we should exploit more.
    # But the directive says: "intensifies as the function value approaches zero"
    # So beta should increase as f_best -> 0
    # beta = beta_base / f_best_abs would blow up as f_best -> 0
    # Let's use: beta = beta_base * (1 + 1/f_best_abs) or similar
    
    # A more stable formulation:
    # beta = beta_base / (1 + f_best_abs)
    # When f_best is large (early), beta is small (exploit)
    # When f_best is small (late), beta is larger (explore)
    # Wait, that's backwards. Early on we should explore more.
    
    # Let me reconsider. In standard GP-LCB:
    # beta_t = sigma^2 * log(t^2 * pi^2 / (6 * delta))
    # This increases with t (number of iterations).
    
    # For cost-aware: divide by cost
    # score = (-mu + beta * sigma) / cost
    
    # Let's use a simple dynamic beta:
    # beta increases as we get closer to the optimum to ensure we don't get stuck
    # But actually, we want to exploit when we're close.
    
    # Let's try: beta = beta_base * (1 + 1 / (f_best_abs + 1e-3))
    # This gives higher beta when f_best is small
    
    beta = beta_base * (1.0 + 1.0 / (f_best_abs + 1e-3))
    
    # Cost-aware LCB score (higher is better)
    # For minimization: we want low mu, and we add exploration bonus
    # score = (-mu + beta * sigma) / cost
    scores = (-mu + beta * sigma) / cost
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
