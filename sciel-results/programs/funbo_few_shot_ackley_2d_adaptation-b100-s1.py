"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses a dynamic Upper Confidence Bound (UCB) for minimization.
    Score = -mu + kappa * sigma.
    Kappa is adjusted based on f_best to balance exploration and exploitation.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)

    # Fallback
    def fallback_lcb():
        kappa = 1.0
        scores = -mu + kappa * sigma
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    if time.time() > deadline:
        return fallback_lcb()

    n = len(mu)
    if n == 0:
        return np.array([])

    # Dynamic Kappa
    # We want more exploration (higher kappa) when f_best is high (far from optimum 0).
    # We want less exploration (lower kappa) when f_best is low (close to optimum 0).
    # f_best for Ackley on [-5, 5] is typically in [0, ~10].
    # Use a smooth transition.
    f_scale = 5.0
    kappa_min = 0.5
    kappa_max = 2.0
    
    # If f_best is 0, kappa = kappa_min.
    # If f_best is large, kappa -> kappa_max.
    # kappa = kappa_min + (kappa_max - kappa_min) * (f_best / (f_scale + f_best))
    
    # Handle potential negative f_best (shouldn't happen for Ackley min 0, but safe)
    f_best_val = max(0.0, float(f_best))
    kappa = kappa_min + (kappa_max - kappa_min) * (f_best_val / (f_scale + f_best_val))

    scores = -mu + kappa * sigma

    # Check deadline
    if time.time() > deadline:
        return fallback_lcb()

    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
# EVOLVE-BLOCK-END
