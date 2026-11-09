"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are too close to the limit
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # Return a simple deterministic score to finish quickly
        # Prefer points with low cost and low mean (optimistic)
        return (-mu + 0.5 * sigma) / np.maximum(cost, 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Clip f_best to [0, 1] for stable interpolation (Griewank range)
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    
    # z-score for Probability of Improvement and Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive beta for UCB-like exploration component
    # Higher beta when f_best is poor (early stages), lower when good (late stages)
    beta_min = 0.3
    beta_max = 1.5
    beta = beta_min + (beta_max - beta_min) * f_best_clipped
    
    # UCB score for minimization: -mu + beta * sigma
    # This represents the "optimistic" potential of the point
    ucb_raw = -mu + beta * sigma
    
    # We want to combine PI, EI, and UCB.
    # PI is bounded [0, 1] and provides a stable exploration signal.
    # EI is unbounded and provides aggressive exploitation.
    # UCB provides a balance.
    
    # Strategy:
    # Early on (high f_best), rely more on PI (stable exploration) and UCB (uncertainty).
    # Later (low f_best), rely more on EI (exploitation).
    
    # Normalize EI to be comparable to PI?
    # EI can be large if f_best is large. But f_best is in [0, 1].
    # Let's use a weighted sum.
    
    # Weights:
    # w_pi: high early, low late
    # w_ei: low early, high late
    # w_ucb: moderate throughout
    
    w_pi = 0.5 * f_best_clipped + 0.1  # ~0.6 early, ~0.1 late
    w_ei = 0.5 * (1.0 - f_best_clipped) + 0.1  # ~0.1 early, ~0.6 late
    w_ucb = 0.4  # constant moderate weight
    
    # Normalize weights to sum to 1? Or just use as is?
    # Using as is is fine, but let's ensure they are positive.
    
    # Scores:
    score_pi = pi / cost
    score_ei = ei / cost
    score_ucb = ucb_raw / cost
    
    scores = w_pi * score_pi + w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
