"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _ei_per_cost(mu, sigma, f_best, cost, xi=0.0, beta=1.0):
    """Expected improvement per unit cost with optional exploration bonus."""
    eps = 1e-9
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma_safe
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip negative EI to zero
    ei = np.maximum(ei, 0.0)
    # Cost-aware: divide by cost
    score = ei / np.maximum(cost, eps)
    # Add small exploration bonus for high uncertainty with low cost
    # This encourages exploring uncertain regions cheaply
    uncertainty_bonus = beta * sigma_safe / np.maximum(cost, eps)
    # Blend: use EI primarily, but add a bit of exploration
    # Actually, let's keep it pure EI/cost for now, but maybe add a small
    # penalty for very low EI to avoid numerical issues? No, keep it clean.
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Compute EI per unit cost
    scores = _ei_per_cost(mu, sigma, f_best, cost, xi=0.0, beta=0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: add a tiny bit of noise to break ties? No, keep deterministic.
    # But we should ensure no negative scores (EI is non-negative, cost positive)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
