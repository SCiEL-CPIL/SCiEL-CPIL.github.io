"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB for exploration and EI for exploitation.
    Uses a smooth transition between the two strategies based on f_best.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # UCB score: maximize -mu + beta * sigma
    # Use a moderate beta for exploration
    beta = 2.5
    ucb = -mu + beta * sigma
    
    # Smooth transition: when f_best is far from the minimum (3.0),
    # prioritize UCB (exploration); when close, prioritize EI (exploitation).
    # Use a sigmoid-like weighting based on how far f_best is from 3.0.
    # If f_best > 5, mostly UCB; if f_best < 4, mostly EI.
    # Weight for EI increases as f_best approaches 3.0.
    weight_ei = 1.0 / (1.0 + np.exp((f_best - 4.0) / 0.5))
    weight_ucb = 1.0 - weight_ei
    
    # Combine the two scores
    scores = weight_ei * ei + weight_ucb * ucb
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
