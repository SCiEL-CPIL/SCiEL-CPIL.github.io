"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition function score for Bayesian Optimization.
    Uses a hybrid of Expected Improvement and Probability of Improvement with
    a dynamic uncertainty floor, normalized before cost division.
    
    Args:
        mu: Array of predicted means (shape 2000,).
        sigma: Array of predicted standard deviations (shape 2000,).
        f_best: The lowest observed function value.
        cost: Array of evaluation costs for each candidate (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,), higher is better.
    """
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Dynamic exploration floor based on f_best
    # f_best_norm: 0 when f_best is best (~-4), 1 when f_best is worst (~0)
    f_best_norm = np.clip((f_best + 4.0) / 4.0, 0.0, 1.0)
    
    # Exploration weight: higher when f_best is worse (early), lower when f_best is better (late)
    # This ensures early exploration and late exploitation
    exploration_weight = 0.1 + 0.3 * f_best_norm
    
    # Uncertainty bonus: normalized sigma
    max_sigma = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / max_sigma
    
    # Blended acquisition: EI + exploration_weight * PI * sigma_norm
    # This combines exploitation (EI) with exploration (uncertainty-weighted PI)
    blended = ei + exploration_weight * pi * sigma_norm
    
    # Normalize blended scores
    blended_max = np.max(blended) if np.max(blended) > 0 else 1.0
    blended_norm = blended / blended_max
    
    # Cost-aware: divide by cost
    scores = blended_norm / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
