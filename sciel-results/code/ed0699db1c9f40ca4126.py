"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function with
    adaptive exploration and improvement-aware weighting.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: scalar - best observed value
        cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the standard LCB: lower is better
    # Use a base beta that balances exploration and exploitation
    beta_base = 2.0
    
    # Adaptive beta: increase exploration when we're far from the optimum
    # or when uncertainty is high relative to the current best
    # Normalize uncertainty by the scale of the problem
    mu_scale = np.maximum(np.abs(mu), 1e-6)
    sigma_ratio = sigma_safe / (np.abs(f_best) + 1e-6)
    
    # Increase beta when uncertainty is high (relative to problem scale)
    # This encourages more exploration in uncertain regions
    beta_adaptive = beta_base * (1.0 + 0.5 * np.clip(sigma_ratio, 0, 2.0))
    
    # Compute LCB values
    lcb = mu - beta_adaptive * sigma_safe
    
    # Improvement-aware weighting: candidates that are predicted to be
    # significantly better than f_best get a boost
    # This helps focus on regions with high expected improvement
    improvement = np.maximum(f_best - mu, 0)
    improvement_weight = 1.0 + 0.3 * improvement / (np.abs(f_best) + 1e-6)
    
    # Combine: we want to maximize the acquisition score
    # Score = -(LCB * improvement_weight) / cost
    # This is equivalent to (beta * sigma - mu) * weight / cost
    scores = (beta_adaptive * sigma_safe - mu) * improvement_weight / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
