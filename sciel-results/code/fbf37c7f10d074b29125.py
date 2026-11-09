"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: track elapsed time to ensure we finish within budget
    start_time = time.time()
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This naturally balances exploration (high sigma) and exploitation (low mu)
    
    # Compute z-score, handling near-zero sigma safely
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Compute normal PDF and CDF
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Add a small exploration bonus to help escape local optima
    # Use a moderate coefficient to not overwhelm the EI signal
    # The bonus is proportional to sigma but scaled down
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    if sigma_max < 1e-10:
        sigma_max = 1.0
    
    # Small exploration bonus: 0.1 * normalized sigma
    exploration_bonus = 0.1 * (sigma / sigma_max)
    
    # Combined score: EI + small exploration bonus
    score = ei + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check: if we're taking too long, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
