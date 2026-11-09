"""Cost-aware acquisition for rastrigin 2d BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Uses Thompson Sampling: draw samples from the GP posterior,
    score each candidate by the expected minimum cost-normalized value
    across samples.
    """
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Number of MC samples for Thompson Sampling
    n_samples = 20
    
    # Draw samples from the posterior: mu + sigma * N(0,1)
    # Shape: (n_samples, 2000)
    rng = np.random.default_rng()
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, len(mu)))
    
    # For each sample, compute the cost-normalized objective: sample / cost
    # We want to minimize this, so we take the negative for maximization
    # Score for each candidate = -mean over samples of (sample_value / cost)
    # But we need to be careful: we want to pick candidates that have low
    # expected cost-normalized value
    
    # Compute cost-normalized values: (n_samples, 2000)
    normalized_samples = samples / cost_safe[None, :]
    
    # For Thompson Sampling, we want to pick the candidate with the lowest
    # expected value. So we compute the mean of the negative normalized values
    # across samples for each candidate
    # score = -mean(normalized_samples, axis=0)
    
    # But to balance exploration, we can also consider the variance
    # A candidate with high variance might be better for exploration
    # Let's use: score = -mean + alpha * std, where alpha controls exploration
    
    mean_norm = np.mean(normalized_samples, axis=0)
    std_norm = np.std(normalized_samples, axis=0)
    
    # Adaptive exploration: more exploration when f_best is large
    f_best_clipped = np.clip(f_best, 0.0, 50.0)
    alpha = 0.3 * np.exp(-f_best_clipped / 10.0)
    
    score = -mean_norm + alpha * std_norm
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
