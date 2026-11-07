"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a deterministic, cost-normalized
    Lower Confidence Bound (LCB) that maximizes the optimistic gap
    (f_best - mu + beta * sigma) / cost.
    
    beta is dynamically scaled based on the current uncertainty budget to
    force aggressive exploration of high-variance, low-cost regions in the
    multimodal Rastrigin landscape.
    """
    # Sanitize inputs to prevent numerical issues
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Dynamic beta scaling based on uncertainty budget
    # Use the mean sigma as a proxy for remaining uncertainty
    mean_sigma = np.mean(sigma)
    
    # Scale beta: higher uncertainty -> more exploration
    # Beta ranges from ~2 to ~10 based on relative uncertainty
    # When f_best is close to 0 (minimum), we want more exploration
    # When f_best is large, we can be more exploitative
    if f_best > 0:
        # Normalize f_best: 0 means at minimum, large values mean far from minimum
        # Rastrigin 2D max is around 40, so normalize by 10 for scaling
        f_norm = min(f_best / 10.0, 1.0)
        # When f_best is small (close to 0), use higher beta for exploration
        # When f_best is large, use lower beta for exploitation
        beta_base = 4.0 + 6.0 * (1.0 - f_norm)
    else:
        beta_base = 10.0
    
    # Additional scaling based on mean uncertainty
    # Higher mean sigma -> slightly higher beta
    sigma_scale = 1.0 + 0.5 * min(mean_sigma / max(np.std(sigma), 1e-6), 2.0)
    beta = beta_base * sigma_scale
    
    # Compute LCB: f_best - mu + beta * sigma
    # This is the optimistic estimate of how much we can improve
    lcb = f_best - mu + beta * sigma
    
    # Ensure LCB is non-negative (we only care about potential improvement)
    lcb = np.maximum(lcb, 0)
    
    # Cost-normalized score
    scores = lcb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
