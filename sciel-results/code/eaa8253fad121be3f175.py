"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    # Probability of Improvement (PI) for minimization
    # We want P(f < f_best - xi)
    # z = (f_best - xi - mu) / sigma
    # Using a small xi to encourage exploration slightly beyond the current best
    # This is generally more robust than EI for small sample sizes (n=5 initial)
    # because it doesn't heavily penalize high uncertainty unless mu is also high.
    
    xi = 0.01
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - xi - mu) / sigma_safe
    
    # PI = Phi(z)
    pi_scores = norm.cdf(z)
    
    # Add a small amount of noise to break ties and encourage exploration
    # This helps if the GP is very confident (low sigma) but we might be stuck in a local optima
    # The noise is scaled by the uncertainty to be more aggressive in uncertain regions
    # However, standard PI is often sufficient. Let's stick to pure PI first, 
    # but the "random tie-breaker" idea suggests adding noise.
    # A common variant is PI + epsilon * noise.
    # Let's use a small constant noise to ensure finite scores and diversity.
    noise = 0.01 * np.random.rand(len(mu))
    
    scores = pi_scores + noise
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
