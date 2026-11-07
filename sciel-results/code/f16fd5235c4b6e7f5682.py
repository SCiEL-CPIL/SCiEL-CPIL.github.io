"""Rewritten acquisition function for Bayesian optimization of 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = 295.0  # seconds, safety margin before 300s kill

    # Ensure inputs are numpy arrays with correct dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Prevent division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute z-scores for Expected Improvement (Minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical issues at extremes
    z_clipped = np.clip(z, -10.0, 10.0)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # Add a small exploration bonus to encourage diversity in high-uncertainty regions
    # This helps escape local minima in multimodal landscapes
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus

    # Check deadline
    if time.time() - start_time > deadline - 10:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return scores

    return scores
# EVOLVE-BLOCK-END
