"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    t0 = time.time()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Optional: Blend with EI?
    # Let's try pure PI first as it is a distinct, strong alternative.
    # If we want to be safe, we can use EI.
    # Let's try a hybrid: EI is generally better. 
    # Let's stick to EI but optimize the implementation.
    
    # Actually, let's try a "sharp" EI or PI.
    # Let's use PI.
    
    score = pi
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > 290.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
