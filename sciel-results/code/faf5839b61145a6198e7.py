"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time to guard against timeout
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 300s limit, 10s safety margin


def acquisition(mu, sigma, f_best, cost):
    global _start_time, _DEADLINE
    
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        # Fallback: use simple PI/cost to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = 0.5 * (1.0 + np.sign(z) * np.sqrt(1.0 - np.exp(-z * z)))  # approx CDF
        # Actually use a simpler approximation
        pi = np.clip(0.5 + 0.5 * z / np.sqrt(1 + z * z / 2), 0, 1)
        score = pi / cost
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Thompson Sampling: sample from posterior and pick lowest cost-normalized value
    # This is a stochastic exploration strategy that differs from deterministic EI/PI
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random samples from the GP posterior
    # For Thompson sampling, we sample a function value for each candidate
    # and select the one with the best (lowest) value per unit cost
    rand = np.random.randn(len(mu))
    noisy_mu = mu - sigma_safe * rand  # Minimization: lower is better, so subtract
    
    # Score: lower noisy_mu / cost is better, but we need higher = evaluate next
    # So we invert: score = -noisy_mu / cost, or use negative
    score = -noisy_mu / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
