"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds of safety margin
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a posterior sampling-based estimate of
    Expected Improvement, blended with an adaptive UCB term.
    
    We draw 50 samples from the GP posterior at the candidate points and
    compute the maximum improvement over f_best across these samples as a
    stochastic proxy for the true expected improvement. This is blended
    with a deterministic UCB term that decays over time.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    global _call_count
    
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 2.0 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    _call_count += 1
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive parameters based on evaluation count
    # We have 30 total evaluations (5 initial + 25 BO steps)
    # _call_count goes from 1 to 25 for the BO steps
    # Progress: 0 at first BO step, 1 at last BO step
    n_total = 25  # number of BO acquisition calls expected
    progress = min(_call_count / n_total, 1.0)
    
    # Beta decays from 2.5 (early, high exploration) to 1.0 (late, exploitation)
    beta = 2.5 - 1.5 * progress
    
    # Blend weight: early on favor posterior sampling (w_ps high), later favor UCB
    # w_ps goes from 0.6 to 0.4
    w_ps = 0.6 - 0.2 * progress
    w_ucb = 1.0 - w_ps
    
    # Number of posterior samples
    n_samples = 50
    
    # Draw samples from the GP posterior: f_i ~ N(mu_i, sigma_i^2)
    # samples has shape (n_samples, N)
    samples = mu[np.newaxis, :] + sigma[np.newaxis, :] * np.random.randn(n_samples, len(mu))
    
    # For each sample, compute the improvement over f_best: max(0, f_best - f_sample)
    # improvement has shape (n_samples, N)
    improvement = np.maximum(0.0, f_best - samples)
    
    # Posterior sampling acquisition: mean of max improvement across samples
    # This approximates the expected improvement
    ps_score = np.mean(improvement, axis=0)
    
    # UCB component: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Combine posterior sampling score and UCB with adaptive weights
    scores = w_ps * ps_score + w_ucb * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
