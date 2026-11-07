"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Primarily exploitation-focused acquisition function using Expected Improvement,
    with a minimal conditional exploration term for very uncertain regions.
    
    For minimization, EI is the expected improvement over f_best:
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    A small exploration bonus is added only where sigma is exceptionally high 
    relative to the scale of mu, to avoid completely ignoring unexplored areas
    while maintaining strong exploitation focus.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    t0 = time.time()
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # d = (f_best - mu) / sigma
    d = (f_best - mu) / sigma_safe
    
    # Use stable computation for EI
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    ei = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Conditional exploration: only add a small bonus where uncertainty is very high
    # relative to the range of mu values. This prevents the acquisition from being
    # purely greedy while maintaining strong exploitation.
    mu_range = np.ptp(mu)
    if mu_range < 1e-12:
        mu_range = 1.0
    
    # Scale sigma by mu_range to get a relative uncertainty measure
    sigma_relative = sigma_safe / mu_range
    
    # Only explore where relative uncertainty exceeds a high threshold
    # This is a very conservative exploration strategy
    explore_threshold = 0.5
    explore_mask = sigma_relative > explore_threshold
    
    # Small exploration bonus: proportional to how much above threshold
    explore_bonus = np.zeros_like(ei)
    explore_bonus[explore_mask] = 0.1 * (sigma_relative[explore_mask] - explore_threshold)
    
    # Combine: primarily EI with small conditional exploration
    scores = ei + explore_bonus
    
    # Ensure all values are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 5.0:
        pass  # Computation is already done, just return
    
    return scores
# EVOLVE-BLOCK-END
