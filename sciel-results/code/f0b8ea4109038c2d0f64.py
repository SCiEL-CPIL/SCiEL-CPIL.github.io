"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
GLOBAL_DEADLINE = None


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition with cost-awareness.
    
    We sample one posterior sample theta = mu + sigma * z, then compute
    a normalized regret score: - (theta - f_best) / sqrt(cost)
    Higher score = more likely to be a good next evaluation given cost.
    
    We also add a small UCB-like exploration term to avoid collapsing
    to a single point early.
    """
    # Ensure deadline guard is set
    global GLOBAL_DEADLINE
    if GLOBAL_DEADLINE is None:
        # Set deadline to 300s from start (with safety margin)
        # We don't know when the evaluator started, so use a conservative approach:
        # just make sure we return quickly. The 2000-element ops are fast.
        GLOBAL_DEADLINE = time.time() + 295.0  # 5s safety margin
    
    n = len(mu)
    
    # Draw a single standard normal vector for Thompson Sampling
    z = np.random.standard_normal(n)
    
    # Posterior sample
    theta = mu + sigma * z
    
    # Cost-aware normalized regret: we want low theta, and we penalize high cost
    # Score: higher is better for selection
    # Use -theta / cost_weight where cost_weight captures the cost penalty
    # The cost is 1 + mean(x_scaled) which is in [1, 2]
    # Use sqrt(cost) as a mild penalty, or just cost directly
    
    # Thompson Sampling score: we want to maximize probability of improving
    # Equivalent to minimizing theta, adjusted for cost
    # score = -(theta - f_best) / cost  -> higher when theta is much below f_best and cost is low
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary: Thompson sampling normalized by cost
    ts_score = -(theta - f_best) / cost_safe
    
    # Add small exploration bonus for high uncertainty (UCB-like)
    # This helps early on when sigma is large
    exploration = sigma / np.sqrt(cost_safe)
    
    # Blend: primarily Thompson, with some exploration
    alpha = 0.3  # weight for exploration
    score = ts_score + alpha * exploration
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
