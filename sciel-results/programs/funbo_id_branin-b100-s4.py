"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB for exploration and POI for softer exploitation.
    UCB is known to be effective for multi-modal functions like Branin by encouraging
    exploration of unexplored regions. POI provides a rank-based, softer exploitation
    signal compared to linear UCB, potentially balancing the trade-off better.
    
    We combine them with a weighted sum. Since we don't know the step count, we use
    fixed weights that favor exploration (UCB) but include some exploitation (POI).
    """
    # Track time to ensure we don't exceed the 300s limit per call (though this is fast)
    t_start = time.time()
    
    # Safety margin for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is between 0 and 1. Higher is better.
    poi = norm.cdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Lower mu is better, higher sigma is better.
    # Score_ucb = -mu + beta * sigma
    # Let's use beta = 1.5, which was effective in the inspiration (score 1.42)
    # The inspiration used beta=1.0 and got 1.42. Let's stick with a robust value.
    # Actually, the inspiration 1 used beta=1.0 and scored 1.4206.
    # Let's try beta=1.0 for UCB component.
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Normalize both to [0, 1] range to combine them fairly
    # POI is already in [0, 1]
    
    # UCB can be any real number. We normalize it based on the current candidate set.
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-9:
        ucb_norm = np.ones_like(ucb) * 0.5
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: 50% UCB, 50% POI
    # This balances the strong exploration of UCB with the softer exploitation of POI.
    score = 0.5 * ucb_norm + 0.5 * poi
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time (should be negligible)
    # t_end = time.time()
    # if t_end - t_start > 290: # Safety guard, unlikely to trigger
    #     return score
    
    return score
# EVOLVE-BLOCK-END
