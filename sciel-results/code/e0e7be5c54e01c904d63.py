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

    # Compute z-scores for Expected Improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Check deadline
    if time.time() - start_time > deadline - 10:
        return ei

    # Compute UCB for minimization: -mu + beta * sigma
    # Use a fixed beta for stability, tuned for exploration
    beta = 2.0
    ucb = -mu + beta * sigma_safe

    # Standardize scores to have zero mean and unit variance
    # This makes EI and UCB comparable in scale without clamping to [0, 1]
    def standardize(s):
        s_mean = np.mean(s)
        s_std = np.std(s)
        if s_std < 1e-9:
            return np.zeros_like(s)
        return (s - s_mean) / s_std

    ei_std = standardize(ei)
    ucb_std = standardize(ucb)

    # Combine: weight EI more heavily but keep UCB for exploration
    # Using z-scores allows for a direct linear combination
    combined = 0.8 * ei_std + 0.2 * ucb_std

    # Ensure all scores are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return combined

    return combined
# EVOLVE-BLOCK-END
