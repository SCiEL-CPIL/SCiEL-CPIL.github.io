# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # Larger values indicate more promising points (lower predicted mean + uncertainty)
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Normalize both to [0, 1] range for combination
    def min_max_normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-15:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = min_max_normalize(ei)
    ucb_norm = min_max_normalize(ucb)
    
    # Combine: 60% exploitation (EI), 40% exploration (UCB)
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
