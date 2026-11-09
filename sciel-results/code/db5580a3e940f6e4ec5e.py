"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        # Simple greedy: minimize mu, penalize cost
        return -mu / (cost + 1e-9)

    epsilon = 1e-9
    
    # Cost-aware Expected Improvement with adaptive exploration
    # Key insight: cost = 1 + mean(x_scaled), so lower cost means closer to origin
    # We want to balance improvement potential against evaluation cost
    
    # 1. Expected Improvement for minimization
    z = (f_best - mu) / (sigma + epsilon)
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # 2. Adaptive cost weighting
    # Normalize cost to [0, 1] range for this batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + epsilon
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # 3. Uncertainty-aware exploration bonus
    # Higher sigma means more potential for improvement in unexplored regions
    sigma_max = np.max(sigma) + epsilon
    sigma_norm = sigma / sigma_max  # 0 to 1
    
    # 4. Combined score: EI dominates, but uncertainty in cheap regions gets a boost
    # The cost penalty is nonlinear: expensive points are heavily penalized
    cost_penalty = 1.0 / (1.0 + 2.0 * cost_norm)  # Cheaper points get higher weight
    
    # Exploration bonus: encourage exploring uncertain cheap regions
    exploration_bonus = sigma_norm * cost_penalty * 0.5
    
    # Exploitation: pure EI weighted by cost efficiency
    exploitation = ei * cost_penalty
    
    # Final score: balance exploitation and exploration
    score = exploitation + exploration_bonus
    
    # Ensure output is finite
    score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return score
# EVOLVE-BLOCK-END
