"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are extremely close to the limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback: return -mu (exploit current best) to ensure finite output
        return -np.asarray(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robustness: ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Structural Change: Use Upper Confidence Bound (UCB) instead of Expected Improvement (EI).
    # EI is prone to local optima in multimodal functions like Branin 2D if the GP is overconfident.
    # UCB explicitly balances exploitation (low mu) and exploration (high sigma).
    # For minimization, we minimize mu - beta * sigma.
    # The score (higher is better) is therefore: -mu + beta * sigma.
    
    # Beta controls the trade-off. 
    # A value of 2.0-3.0 is often effective for Branin to escape local minima.
    # We choose 2.5 to be more exploratory than standard EI.
    beta = 2.5
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all values are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
