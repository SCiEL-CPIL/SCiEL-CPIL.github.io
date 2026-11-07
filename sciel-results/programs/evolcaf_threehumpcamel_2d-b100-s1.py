"""
Strategy: Model the problem as a multi-objective optimization where the primary objective is the negative function value and the secondary objective is the evaluation cost, using a hypervolume-based or weighted-sum Pareto front approximation to score candidates, thereby explicitly trading off improvement magnitude against computational expense rather than simple division.
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
        # Just return EI / cost as a fallback
        sigma_safe = np.maximum(sigma, 1e-6)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-6)
        return np.nan_to_num(ei / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1.0) # Cost is 1 + mean(x_scaled), so min is 1
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard EI for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If f_best - mu is large positive, EI is large.
    # If f_best - mu is negative (mu < f_best), EI is still positive due to the second term if sigma is large enough,
    # but effectively 0 if we clip. Actually, EI is always >= 0.
    improvement = f_best - mu
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Multi-objective / Pareto-inspired scoring
    # We want to maximize EI while minimizing Cost.
    # Simple division EI/Cost is a specific scalarization.
    # Let's try a weighted sum in the "objective space" where objectives are (EI, -Cost) or (EI, Cost).
    # Since we want to maximize EI and minimize Cost, we can define a score that balances them.
    
    # Normalization:
    # EI varies significantly. Cost is in [1, 2].
    # Let's normalize EI by the maximum EI in the batch to get a relative improvement score in [0, 1].
    max_ei = np.max(ei)
    if max_ei > 1e-10:
        ei_norm = ei / max_ei
    else:
        ei_norm = ei
        
    # Cost is in [1, 2]. Let's normalize it to [0, 1] where 1 is most expensive.
    # cost_norm = (cost - 1) / (2 - 1) = cost - 1
    cost_norm = np.clip(cost_safe - 1.0, 0.0, 1.0)
    
    # Pareto front approximation via weighted sum:
    # We want a convex combination that favors exploration when EI is high, but penalizes cost.
    # A common approach in multi-objective BO is to use the hypervolume contribution, but that's expensive.
    # A simpler alternative is the "Weighted Sum" with adaptive weights.
    
    # Let's try: Score = ei_norm * w_ei + (1 - cost_norm) * w_cost
    # Where w_ei + w_cost = 1.
    # If we set w_ei = 0.7, w_cost = 0.3, we prioritize improvement but care about cost.
    # However, EI is already an absolute measure. Normalizing it to max=1 might lose scale info.
    
    # Alternative: Use the "Probability of Improvement" (PI) as a secondary signal?
    # PI = Phi(z). It is in [0, 1].
    pi = norm.cdf(z)
    
    # Let's try a different scalarization:
    # Score = EI * (1 - alpha * cost_norm)
    # This reduces the effective EI for expensive points.
    # alpha = 0.5 means a point with max cost (cost_norm=1) gets half the EI score.
    alpha = 0.5
    scores = ei * (1.0 - alpha * cost_norm)
    
    # To make it more "Pareto-like", we can also add a small bonus for high PI if EI is low?
    # No, let's stick to the weighted sum of normalized objectives.
    
    # Let's try the normalized weighted sum approach again, but keep the scale of EI.
    # Score = (EI / Max_EI) * 0.8 + ((2 - Cost) / (2 - 1)) * 0.2
    # Here (2-Cost)/1 is in [0,1], higher is cheaper.
    # This ensures the score is bounded and comparable.
    
    # But wait, if Max_EI is very small, EI/Max_EI is noisy.
    # Let's use a robust normalization: EI / (Max_EI + epsilon)
    
    # Actually, the prompt suggests "hypervolume-based or weighted-sum Pareto front approximation".
    # Let's implement a simple weighted sum of the two objectives:
    # Objective 1: Maximize EI
    # Objective 2: Minimize Cost
    
    # Normalize both to [0, 1] range for fair weighting.
    # EI_norm is already [0, 1].
    # Cost_inv_norm = (2 - cost) / (2 - 1) = 2 - cost. Since cost in [1, 2], this is [0, 1].
    cost_inv_norm = 2.0 - cost_safe
    
    # Weights:
    # Early on, we might want to explore more (higher EI relative to cost).
    # Later, we want to exploit (higher EI, cost matters less? Or cost matters more to save budget?).
    # Since the budget is on total cost, saving cost is always good.
    # Let's use fixed weights for simplicity and stability: 70% EI, 30% Cost.
    w_ei = 0.7
    w_cost = 0.3
    
    scores = w_ei * ei_norm + w_cost * cost_inv_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
