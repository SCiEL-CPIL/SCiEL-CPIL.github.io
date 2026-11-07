"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient acquisition with sublinear cost discount.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - best observed function value
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Knowledge Gradient: E[max(f_best, f(x))] - f_best
    # This equals E[(f_best - f(x))^+] which is the Expected Improvement
    # but we use a different scaling that emphasizes uncertainty more
    
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # KG-like adjustment: weight by probability that x improves on f_best
    # and add a stronger exploration component based on the ratio of uncertainty to current best
    p_improve = norm.cdf(z)
    
    # Scale factor that emphasizes points with high uncertainty relative to the gap
    # When mu is close to f_best (small gap), high sigma means high KG value
    gap = np.maximum(f_best - mu, 0)
    uncertainty_weight = sigma_safe / (gap + sigma_safe + 1e-9)
    
    # Combined: EI weighted by uncertainty importance, plus exploration bonus
    kg_score = ei * (1 + uncertainty_weight)
    
    # Add a small exploration bonus for very uncertain points
    exploration_bonus = 0.5 * sigma_safe * (1 - p_improve)
    
    scores = (kg_score + exploration_bonus) / np.sqrt(cost_safe)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
