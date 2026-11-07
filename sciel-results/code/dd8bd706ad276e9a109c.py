"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We'll use a time guard to ensure we finish well before the 300s limit.
# The acquisition function is called once per iteration (2000 candidates scored),
# so we need it to be fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means more likely to be selected next.
    """
    # Sanitize inputs to ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # f_best should be a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -50, 50)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative (it should be by definition, but numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Also add a small exploration bonus based on uncertainty to encourage exploring
    # high-uncertainty regions, especially early on when f_best is far from the minimum.
    
    # A simple approach: blend EI/cost with a scaled uncertainty term
    # The uncertainty term encourages exploration in areas with high sigma.
    # We scale it by a factor that depends on how far f_best is from the known minimum (0).
    
    # For rastrigin, the minimum is 0. So the "gap" is f_best - 0 = f_best.
    # When f_best is large, we should explore more.
    
    # Let's use: score = EI / cost + alpha * sigma / cost
    # where alpha is a small positive constant.
    
    # Actually, a cleaner approach: use a weighted combination.
    # The original EI/cost is a good baseline. Let's add a small exploration bonus.
    
    # Alternative: use a parameterized blend
    # score = (1 - beta) * EI/cost + beta * sigma/cost
    # where beta is small, like 0.1
    
    # But we don't know how many iterations we're on. Let's keep it simple.
    
    # Let's try: score = EI / cost + 0.1 * sigma / cost
    # This adds a small exploration bonus proportional to uncertainty.
    
    exploration_bonus = 0.1 * sigma / cost
    scores = ei / cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
