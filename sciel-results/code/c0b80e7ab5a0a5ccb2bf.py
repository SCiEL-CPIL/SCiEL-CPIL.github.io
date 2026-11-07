"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    if time.time() - _start_time > 325:
        return -mu

    t = _call_count
    n = len(mu)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Structural Change: Use Upper Confidence Bound (UCB) with adaptive exploration parameter beta.
    # UCB balances exploitation (low mu) and exploration (high sigma).
    # For minimization, we maximize: score = -mu + beta * sigma
    
    # Adaptive beta: Start with high exploration to find global minima, decay to exploitation.
    # t starts at 1.
    # At t=1, beta = 1.0 + 5.0/2 = 3.5
    # At t=25, beta = 1.0 + 5.0/26 ~ 1.19
    beta = 1.0 + 5.0 / (1.0 + t)
    
    # Calculate UCB scores
    # We want to maximize this score.
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
