"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 in 8D.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - lowest observed value
    - cost: array of shape (2000,) - cost of evaluating each point (1 + mean(x_scaled))
    
    Returns:
    - score: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    t_start = time.time()
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower bound on objective: -0.8 (minimum of cosine8)
    f_min_possible = -0.8
    
    # Estimate how close we are to the optimum
    improvement_room = f_best - f_min_possible
    
    # Use a softer cost penalty: k = 0.3
    # cost=1: exp(-0.3) ≈ 0.74
    # cost=2: exp(-0.6) ≈ 0.55
    k = 0.3
    
    # Multiplicative gate
    gate = np.exp(-k * cost)
    
    # Base score from EI
    score = ei * gate
    
    # Add a small UCB-style exploration term
    # This encourages exploring high-uncertainty regions but is cost-aware
    # Beta controls the trade-off; start with 0.05 for gentle exploration
    beta = 0.05
    ucb_term = beta * sigma_safe * gate
    
    score = score + ucb_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
