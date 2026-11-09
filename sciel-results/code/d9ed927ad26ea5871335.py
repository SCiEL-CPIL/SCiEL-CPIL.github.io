import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling + Expected Improvement acquisition function
    for Bayesian optimization of sphere 1D minimization.
    
    Combines:
    1. Thompson Sampling: sample from posterior to get exploration
    2. Exploitation term: prefer points with low predicted mean
    3. Uncertainty bonus: small bonus for high uncertainty to encourage exploration
    
    The key insight for sphere 1D is that the minimum is at x=0 with f=0.
    We want to balance finding low-mu regions (exploitation) with exploring
    uncertain regions (exploration).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: pure exploitation
        return -mu.copy()
    
    # Handle near-zero sigma to avoid issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a single sample from the posterior at each candidate
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Thompson score: negative of sampled value (higher = better)
    ts_score = -samples
    
    # Exploitation score: negative of predicted mean
    exploit_score = -mu
    
    # Exploration bonus: proportional to uncertainty, but bounded
    # Scale exploration relative to the range of mu to make it adaptive
    mu_range = mu.max() - mu.min() + 1e-8
    exploration_bonus = 0.3 * sigma_safe / mu_range
    
    # Combined score: weighted combination
    # Weight exploitation heavily since we know the function is smooth
    # and the minimum is well-defined
    alpha = 0.7  # Weight for Thompson sampling
    beta = 0.2   # Weight for pure exploitation
    gamma = 0.1  # Weight for exploration bonus
    
    scores = alpha * ts_score + beta * exploit_score + gamma * exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
