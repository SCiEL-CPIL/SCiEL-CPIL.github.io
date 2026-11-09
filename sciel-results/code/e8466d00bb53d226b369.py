"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Structural Change: Normalized Improvement Heuristic
    # Score = (f_best - mu) / (sigma + epsilon)
    # This encourages points with high potential improvement relative to their uncertainty.
    # It is a deterministic, parameter-free heuristic that balances exploration and exploitation.
    epsilon = 1e-8
    improvement = f_best_safe - mu_safe
    uncertainty = sigma_safe + epsilon
    
    # The score should be higher for better candidates.
    # Since we are minimizing, lower mu is better.
    # improvement = f_best - mu. If mu < f_best, improvement is positive.
    # We want to maximize this ratio.
    final_scores = improvement / uncertainty
    
    # Check time
    if time.time() - start_time > deadline - start_time:
        # Fallback to EI if something went wrong (unlikely)
        z = (f_best_safe - mu_safe) / sigma_safe
        z = np.clip(z, -10, 10)
        ei_scores = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        final_scores = ei_scores

    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
