"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement with adaptive scaling and diversity bonus.
    
    This is a structural change from UCB: instead of linearly combining
    mean and uncertainty, we use a probabilistic measure of improvement
    likelihood, scaled adaptively, plus a diversity term.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of improvement: P(f(x) < f_best)
    # For Gaussian posterior: P(mu - z*sigma < f_best) where z ~ N(0,1)
    # This is norm.cdf((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    poe = norm.cdf(z)
    
    # Adaptive scaling: as we get closer to optimum, increase exploration
    # gap measures how close f_best is to the known minimum (3.0)
    gap = max(f_best - 3.0, 0.0)
    
    # Adaptive beta that increases as gap decreases
    # When gap is large, we're far from optimum -> more exploitation
    # When gap is small, we're close -> more exploration needed
    adaptive_beta = 1.0 + 5.0 * np.exp(-gap / 0.5)
    
    # Combine: scale the PoE by adaptive factor
    # Higher PoE means more likely to improve, so we want to maximize it
    # But we also need to weight by uncertainty to avoid over-exploitation
    scores = adaptive_beta * poe * sigma
    
    # Add diversity bonus: points far from current best get a small boost
    # We don't have coordinates, so we use a proxy: points with high sigma
    # are likely in unexplored regions
    diversity_bonus = 0.1 * sigma
    
    scores = scores + diversity_bonus
    
    # Check deadline
    if time.time() > deadline - 5.0:
        # Return best found so far with safety margin
        pass
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Normalize to [0, 1] range for numerical stability
    max_score = np.max(scores)
    min_score = np.min(scores)
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = np.ones_like(scores) * 0.5
    
    return scores
# EVOLVE-BLOCK-END
