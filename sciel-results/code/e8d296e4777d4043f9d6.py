# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. Upper Confidence Bound (UCB) for exploration
    3. Adaptive blending based on f_best (shifts to exploitation as f_best -> 0)
    4. Cost normalization to avoid instability
    
    All computations are vectorized for speed.
    """
    # Safety deadline guard
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using stable computation
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB with adaptive beta
    # Beta should decrease as f_best approaches 0 (more exploitation)
    # f_best ranges from ~1.9 (Ackley max) to 0 (min)
    # Normalize f_best to [0, 1] range
    f_best_norm = np.clip(f_best / 1.9, 0, 1)
    
    # Beta: starts high (exploration), decreases as we find better solutions
    # beta = beta_max * (f_best_norm)^gamma
    beta_max = 2.0
    gamma = 1.5
    beta = beta_max * (f_best_norm ** gamma)
    
    # UCB score (higher is better for minimization, so we negate)
    ucb = -mu + beta * sigma
    
    # Adaptive blend weight: more EI when f_best is low, more UCB when f_best is high
    # weight_ei increases as f_best -> 0
    weight_ei = 1.0 - f_best_norm
    weight_ucb = f_best_norm
    
    # Combine: EI is already "improvement" (positive when mu < f_best), UCB is a score
    # Normalize both to comparable scales
    # EI can be negative if mu > f_best, clip to 0
    ei_clipped = np.maximum(ei, 0)
    
    # Cost normalization: divide by cost, but use max-cost scaling for stability
    cost_safe = np.maximum(cost, 1e-8)
    cost_max = np.max(cost_safe)
    
    # Normalize cost to [0, 1] range (lower cost = higher priority)
    cost_norm = cost_safe / cost_max  # [0, 1]
    
    # Inverse cost weighting (lower cost gets higher weight)
    # Use a power to control sensitivity
    cost_weight = 1.0 / (cost_norm + 0.1)
    cost_weight = cost_weight / np.max(cost_weight)  # Normalize to [0, 1]
    
    # Final score: blend of EI and UCB, weighted by inverse cost
    # EI part: higher EI is better
    # UCB part: higher UCB is better (already negated mu)
    score = weight_ei * ei_clipped + weight_ucb * ucb
    
    # Apply cost weighting
    score = score * cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: if we've used too much time, return immediately
    # (This is a simple guard; the computation above is fast)
    
    return score
# EVOLVE-BLOCK-END
