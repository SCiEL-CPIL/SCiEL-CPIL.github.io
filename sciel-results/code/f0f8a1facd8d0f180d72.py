import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clamp sigma and cost to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Handle z overflow/underflow
    z = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: use a softer penalty than linear
    # Option 1: EI / cost (current)
    # Option 2: EI / cost^0.7 (softer penalty)
    # Option 3: log(EI + eps) - alpha * log(cost)
    # Let's try a combination: primarily EI/cost but with a floor to avoid extreme penalties
    
    # Use EI / cost but with a minimum cost effect
    # This prefers cheap points with good EI, but doesn't completely ignore expensive points with huge EI
    score = ei / cost_safe
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in multi-modal functions
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    total_score = score + exploration_bonus
    
    # Ensure finite output
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score
