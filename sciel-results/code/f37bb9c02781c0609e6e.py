"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with refined EI/PI blending and 
    principled cost adjustment for Hartmann 3D optimization.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Track elapsed time for safety margin
    t_start = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score with clipping to avoid extreme values
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (for minimization: improvement = f_best - f(x))
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Robust normalization using percentile to avoid outlier dominance
    ei_p95 = np.percentile(ei, 95)
    if ei_p95 > 0:
        ei_norm = ei / ei_p95
    else:
        ei_norm = np.zeros_like(ei)
    
    pi_p95 = np.percentile(pi, 95)
    if pi_p95 > 0:
        pi_norm = pi / pi_p95
    else:
        pi_norm = np.zeros_like(pi)
    
    # Clip normalized values to [0, 1]
    ei_norm = np.clip(ei_norm, 0, 1)
    pi_norm = np.clip(pi_norm, 0, 1)
    
    # Blend EI and PI with exploitation bias (0.7 EI, 0.3 PI)
    # Higher EI weight favors exploitation of known good regions
    blended = 0.7 * ei_norm + 0.3 * pi_norm
    
    # Cost-aware adjustment:
    # Cost is in [1, 2] (1 + mean(x) where x in [0,1])
    # Use a power law to make cost penalty more pronounced for expensive points
    # cost_adj = 1 / (cost_safe ** 1.5) gives stronger penalty for high cost
    cost_safe = np.maximum(cost, 1.0)  # Cost is always >= 1
    cost_penalty = 1.0 / (cost_safe ** 1.5)
    
    scores = blended * cost_penalty
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
