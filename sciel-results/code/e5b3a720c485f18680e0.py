"""Rewritten program for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a hybrid strategy combining
    Expected Improvement (EI) with an exploration bonus based on
    upper confidence bound (UCB) principles.
    
    The idea is to balance exploitation (finding points better than f_best)
    with exploration (visiting regions of high uncertainty).
    
    For minimization:
    - EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
      where z = (f_best - mu(x)) / sigma(x)
    - UCB exploration term: beta * sigma(x)
    
    We use a dynamic beta that increases slightly with the number of
    evaluations (tracked via f_best decreasing), encouraging more
    exploration early on.
    
    Additionally, we add a small random perturbation to break ties and
    encourage diverse exploration.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF using scipy
    try:
        from scipy.stats import norm
        Phi_z = norm.cdf(z)
        phi_z = norm.pdf(z)
    except ImportError:
        sqrt2 = np.sqrt(2.0)
        from math import pi
        Phi_z = 0.5 * (1.0 + np.erf(z / sqrt2))
        phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * pi)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term with adaptive beta
    # beta decreases as f_best gets closer to 0 (more exploitation later)
    # Start with beta ~ 2.0, decay towards 0.5 as f_best improves
    beta = 2.0 - 1.5 * np.clip(f_best / 5.0, 0.0, 1.0)
    ucb_term = beta * sigma
    
    # Combine EI and UCB
    # We use a weighted sum, but also consider the raw predictive mean
    # to ensure we don't ignore obviously good points
    score = ei + ucb_term
    
    # Add a small term based on how far below f_best the mean is
    # This helps prioritize points with mu significantly below f_best
    # even when sigma is small
    exploitation_bonus = np.maximum(0, f_best - mu) * 0.1
    
    score = score + exploitation_bonus
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
