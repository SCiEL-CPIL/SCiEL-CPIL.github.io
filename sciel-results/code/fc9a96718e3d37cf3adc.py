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
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for exploration in multimodal landscape
    beta = 0.5
    ucb = mu + beta * sigma
    
    # Combine EI and UCB with adaptive weighting
    # As f_best approaches 0 (optimum), shift more towards exploitation (EI)
    # Early on, balance exploration to find new basins
    f_best_clipped = max(f_best, 1e-6)
    exploitation_weight = min(1.0, f_best_clipped / 10.0)
    exploration_weight = 1.0 - exploitation_weight
    
    # Normalize EI and UCB for fair combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_min = np.min(ucb)
    ucb_range = np.ptp(ucb) if np.ptp(ucb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weighted combination
    combined = exploitation_weight * ei_norm + exploration_weight * ucb_norm
    
    # Cost-aware normalization: prefer cheaper points
    # Use a power to make cost penalty more aggressive
    cost_factor = 1.0 / (cost ** 0.7 + 1e-6)
    
    scores = combined * cost_factor
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
