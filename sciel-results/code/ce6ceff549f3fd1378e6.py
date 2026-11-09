# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a small
    exploitation bias. For minimization:
    - z = (f_best - mu) / sigma
    - EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    We add a small penalty proportional to sigma to counter over-exploration,
    making the acquisition slightly more exploitation-focused.
    """
    # Handle near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Small exploitation bias: subtract a fraction of sigma to prefer
    # lower-uncertainty regions that still show improvement
    # This counters the over-exploration tendency of pure EI
    beta = 0.1
    score = ei - beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
