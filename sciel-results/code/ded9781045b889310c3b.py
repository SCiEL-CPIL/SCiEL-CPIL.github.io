"""
Strategy: A cost-aware acquisition function using a weighted sum of normalized Expected Improvement (EI), inverse Cost, and cost-adjusted Uncertainty (sigma). 
This approach explicitly balances the potential for improvement, the computational efficiency (cost), and the exploration of uncertain regions, with the uncertainty term penalized by cost to favor cheap exploration.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for safety guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize time guard on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Check deadline (330s limit, use 325s for safety)
    if time.time() - _start_time > 325:
        # Return a valid but potentially suboptimal result to avoid timeout
        # Fallback: EI / Cost
        sigma_safe = np.maximum(sigma, 1e-6)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1.0)
        return np.nan_to_num(ei / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    sigma_safe = np.maximum(sigma, 1e-6)
    # Cost is 1 + mean(x_scaled), so min is 1, max is 2.
    # Clip to valid range to prevent division by zero or negative costs
    cost_safe = np.clip(cost, 1.0, 2.0)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    improvement = f_best - mu
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Cost Score
    # We want higher score for lower cost.
    # Inverse cost normalized to [0, 1].
    # C=1 -> 1/C=1 -> Score=1. C=2 -> 1/C=0.5 -> Score=0.
    cost_score = 2.0 / cost_safe - 1.0
    
    # 3. Compute Cost-Adjusted Uncertainty Score
    # Uncertainty (sigma) is good for exploration, but expensive uncertainty is less valuable.
    # Metric: sigma / cost
    unc_metric = sigma_safe / cost_safe
    max_unc = np.max(unc_metric)
    if max_unc > 1e-6:
        unc_score = unc_metric / max_unc
    else:
        unc_score = np.zeros_like(unc_metric)
        
    # 4. Normalize EI
    # Use 99th percentile for robustness against outliers
    ei_ref = np.percentile(ei, 99)
    if ei_ref > 1e-6:
        ei_norm = ei / ei_ref
    else:
        ei_norm = ei
        
    # 5. Weighted Sum
    # Weights:
    # EI: 0.7 (Primary objective)
    # Cost: 0.2 (Important for budget)
    # Uncertainty: 0.1 (Exploration)
    w_ei = 0.7
    w_cost = 0.2
    w_unc = 0.1
    
    scores = w_ei * ei_norm + w_cost * cost_score + w_unc * unc_score
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
