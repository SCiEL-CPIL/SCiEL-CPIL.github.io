"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with a smooth exponential cost penalty that adapts to optimization progress.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions
    sigma : array_like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array_like, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost is in [1, 2] since cost = 1 + mean(x_scaled_to_[0,1])
    # Normalize cost to [0, 1] range for penalty calculation
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Adaptive lambda: as f_best approaches 0 (the minimum), we become more
    # sensitive to cost. When f_best is large, we care less about cost.
    # f_best starts around the rosenbrock value at random points (could be ~10-100)
    # and approaches 0 as we find the minimum.
    # Use a smooth transition: lambda increases as f_best decreases
    f_best_abs = max(abs(f_best), 1e-6)
    # When f_best is small (close to optimum), lambda is large -> strong cost penalty
    # When f_best is large (far from optimum), lambda is small -> weak cost penalty
    # Scale: lambda ranges from ~0.5 to ~5.0
    lambda_cost = 0.5 + 4.5 * (1.0 / (1.0 + f_best_abs))
    
    # Exponential cost penalty: exp(-lambda * cost_norm)
    # This smoothly penalizes high-cost candidates
    cost_penalty = np.exp(-lambda_cost * cost_norm)
    
    # Probability of Improvement (PI):
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is smoother than EI and maintains a gradient toward the valley floor
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Combine PI with cost penalty
    # The product form ensures that even if PI is high, a very high cost
    # will reduce the score, but the exponential makes this smooth
    scores = pi * cost_penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Small positive floor to avoid all-zero scores
    scores = np.maximum(scores, 1e-10)
    
    return scores
# EVOLVE-BLOCK-END
