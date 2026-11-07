"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
DEADLINE = 325.0

def _check_deadline():
    return (time.time() - _start_time) < DEADLINE

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if not _check_deadline():
        # Fallback: simple UCB
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        return -mu + 0.5 * sigma_safe

    # Thompson Sampling Strategy
    # We sample 50 different "worlds" (functions) from the GP posterior.
    # For each world, we find the candidate with the lowest predicted value.
    # The score for a candidate is the frequency with which it was the best in these worlds.
    
    n_candidates = len(mu)
    n_samples = 50  # Number of Monte Carlo samples
    
    # Generate random normal samples for the function values
    # Shape: (n_samples, n_candidates)
    z = np.random.randn(n_samples, n_candidates)
    
    # Predicted values in each sample: y_k(x_i) = mu_i + sigma_i * z_ki
    # mu and sigma are (n_candidates,), z is (n_samples, n_candidates)
    # We want y to be (n_samples, n_candidates)
    y = mu[None, :] + sigma[None, :] * z
    
    # Find the index of the minimum value in each sample (row)
    # best_indices shape: (n_samples,)
    best_indices = np.argmin(y, axis=1)
    
    # Count how often each candidate was the best
    # counts shape: (n_candidates,)
    counts = np.zeros(n_candidates)
    np.add.at(counts, best_indices, 1)
    
    # The score is the probability of being the best
    scores = counts / n_samples
    
    # Add a small deterministic tie-breaker or exploration term?
    # Pure TS is often sufficient. But to ensure we don't get stuck in a local optimum
    # due to finite sampling, we might add a tiny bit of noise or use the expected value
    # of the minimum? No, standard TS is just the frequency.
    
    # However, TS can be unstable if sigma is very small everywhere.
    # If sigma is small, z * sigma is small, so y is close to mu.
    # The argmin will be the candidate with the lowest mu.
    # This is effectively a "Greedy" search with noise.
    
    # To make it more robust, we can blend with a small amount of EI or UCB?
    # The prompt asks for a DIFFERENT strategy. TS is the different strategy.
    # Let's stick to pure TS but ensure numerical stability.
    
    # If all sigmas are zero (or very small), counts will be concentrated on the global min of mu.
    # This is fine.
    
    # One issue: if the GP is overconfident (sigma -> 0), TS becomes greedy.
    # If the GP is underconfident (sigma -> inf), TS becomes random.
    # This is the intended behavior.
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
