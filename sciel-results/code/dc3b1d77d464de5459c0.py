"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Budget-aware acquisition: blends cost-adjusted EI with a Thompson-sampling-like
    stochastic term whose magnitude is modulated by remaining budget.
    
    As budget depletes, we shift toward exploitation (pure cost-adjusted EI).
    With more budget, we add exploration via sampling from the posterior.
    """
    # Guard against division by zero and negative/zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI calculation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-adjusted EI (exploit efficiency)
    ei_cost = ei / np.maximum(cost, 1e-9)
    
    # Estimate remaining budget: total budget is 30, initial 5 points cost ~5*1.5=7.5 avg
    # cost array has shape (2000,) representing cost of each candidate
    # We don't know total spent, but cost values give us a proxy
    # Use mean cost as a proxy for typical evaluation cost
    mean_cost = np.mean(cost)
    
    # Budget-aware temperature:
    # With high remaining budget (low mean_cost relative to total), explore more
    # With low remaining budget (high mean_cost), exploit more
    # Scale: total budget 30, if mean_cost is low, we have many evaluations left
    # Use a simple heuristic: temperature decreases as mean_cost increases
    # Normalize mean_cost to [0, 1] range assuming costs are in [1, 2]
    cost_norm = np.clip((mean_cost - 1.0) / 1.0, 0, 1)
    
    # Temperature: high when budget is plentiful, low when budget is scarce
    # Start with temperature ~1.0 and decay
    temperature = 1.0 - 0.7 * cost_norm
    
    # Thompson sampling-like stochastic term
    # Sample from posterior: mu + sigma * z_sample
    # But we need to keep it deterministic-ish for reproducibility within a single call
    # Use a fixed random seed based on the call context for consistency
    # Actually, for acquisition functions, we should return deterministic scores
    # Instead, use a UCB-like exploration bonus that scales with temperature
    
    # Exploration bonus: beta * sigma / cost (UCB style, cost-adjusted)
    # beta controls exploration-exploitation trade-off
    beta = 0.5 * temperature
    
    # Combined acquisition: cost-adjusted EI + exploration bonus
    exploration_bonus = beta * sigma_safe / np.maximum(cost, 1e-9)
    
    score = ei_cost + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
