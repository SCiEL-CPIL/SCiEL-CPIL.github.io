"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a weighted combination of posterior mean
    (exploitation) and posterior standard deviation (exploration).
    This implements a 'Posterior Mean Ranking' style strategy with a small
    exploration bonus to avoid local optima, favoring aggressive exploitation
    of the current best region.
    
    mu, sigma: shape (2000,)
    f_best: scalar, lowest observed value
    Returns: shape (2000,) scores, higher = evaluate next
    """
    n = len(mu)
    
    # Ensure sigma is non-negative and has a minimum floor
    sigma = np.maximum(sigma, 1e-8)
    
    # Strategy: Minimize mu, but add a small bonus for uncertainty.
    # Score = -mu + beta * sigma
    # We choose beta to be small to prioritize exploitation (low mu).
    # Normalizing terms helps if scales differ significantly, but for GP
    # outputs in objective units, they are often comparable.
    # Let's try a fixed beta first.
    beta = 0.1
    
    final_scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
# EVOLVE-BLOCK-END
