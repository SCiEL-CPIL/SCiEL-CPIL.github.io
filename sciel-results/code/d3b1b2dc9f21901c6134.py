"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Standard Expected Improvement for minimization
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Clip to non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Use Log-EI to improve exploration robustness
    # This is the key structural change from inspiration 2 that scored 1.0
    # Adding a small epsilon to avoid log(0)
    scores = np.log(ei + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
