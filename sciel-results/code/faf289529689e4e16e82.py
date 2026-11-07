"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Deterministic UCB acquisition function for Weierstrass 1D minimization.
    
    Maximizes -mu + kappa * sigma, where kappa is adapted based on f_best.
    Higher kappa encourages exploration of high-uncertainty regions, which is
    critical for the multimodal Weierstrass function. As f_best approaches 0,
    kappa decreases to focus on exploitation.
    
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best (lowest) observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher is better
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa parameters
    k_max = 3.5  # Exploration strength when far from optimum
    k_min = 0.5  # Exploitation strength when near optimum
    scale = 0.5  # Controls the speed of transition
    
    # f_best is the minimum value observed so far. For Weierstrass, min is 0.
    # If f_best is large, we are far from the optimum -> need high kappa.
    # If f_best is close to 0, we are near the optimum -> need low kappa.
    f_best_safe = max(0.0, f_best)
    decay_term = 1.0 - np.exp(-f_best_safe / scale)
    kappa = k_min + (k_max - k_min) * decay_term
    
    # UCB for minimization: maximize -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
