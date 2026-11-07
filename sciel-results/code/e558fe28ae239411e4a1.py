"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Time guard: ensure we return within the 330s limit
    start_time = time.time()
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle the case where sigma is very small (near zero)
    # When sigma is tiny, EI should be max(f_best - mu, 0)
    safe_sigma = np.where(sigma > 1e-10, sigma, 1.0)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we want higher scores to mean "evaluate next", so we use EI directly
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # For points where f_best - mu <= 0 (already better than best), EI should be 0
    # Actually, EI formula naturally handles this: when f_best - mu < 0, 
    # the improvement is negative, but EI is defined as max(0, improvement)
    # Let's use the proper formula:
    # EI = max(f_best - mu, 0) * Phi(z) + sigma * phi(z) when f_best - mu > 0
    # EI = 0 when f_best - mu <= 0
    
    # More robust computation:
    improvement = f_best - mu
    ei = np.where(improvement > 0, improvement * Phi_z + sigma * phi_z, 0.0)
    
    # Add a small exploration bonus based on sigma to encourage exploration
    # This helps escape local optima in the 6D space
    exploration_weight = 0.3
    scores = ei + exploration_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check - if we're running out of time, just return what we have
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
