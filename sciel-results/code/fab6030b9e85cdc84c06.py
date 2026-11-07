"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost (EIC)
    # This is a more principled cost-aware acquisition than LCB
    # EI measures the expected improvement over f_best
    # Dividing by cost gives expected improvement per unit cost
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_clipped
    
    # Compute expected improvement:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # Use scipy.stats.norm for accuracy
    from scipy.stats import norm
    
    # Vectorized computation
    ei = (f_best - mu) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted: expected improvement per unit cost
    # Higher score = more promising (more improvement per cost)
    score = ei / (cost + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
