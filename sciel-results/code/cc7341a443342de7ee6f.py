import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best):
    global _start_time, _DEADLINE
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return a safe fallback: EI
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Thompson Sampling component
    # This adds stochastic exploration on top of the deterministic EI
    # We blend EI with a Thompson sample to get both exploitation and exploration
    z_rand = np.random.randn(2000)
    thompson_sample = mu + sigma_safe * z_rand
    
    # For minimization, higher score = more promising
    # EI is already "higher is better" (positive when likely to improve)
    # Thompson: -sample is higher when sample is lower (better for minimization)
    
    # Blend: primarily EI with a small Thompson exploration component
    # The Thompson component adds variance that encourages exploring uncertain regions
    alpha = 0.3  # Weight for Thompson exploration
    scores = ei + alpha * (-thompson_sample + mu)  # Center the Thompson sample
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
