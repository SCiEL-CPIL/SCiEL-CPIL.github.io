"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Expected Hypar Improvement (EHI) with Gaussian damping for cost-aware BO.
    
    mu, sigma: arrays of shape (2000,) - GP mean and std in objective units
    f_best: float - lowest observed value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative costs
    cost = np.maximum(cost, 1e-6)
    
    # Standard deviation guard
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Hypar Improvement (EHI) for minimization
    # EHI uses a nonlinear saturation: hypar(f, c) = -log(1 + exp(-(f - c)))
    # For minimization with cost awareness, we use:
    # EHI = E[hypar(f(X), cost)] where hypar penalizes high function values
    # and accounts for evaluation cost via the cost parameter
    
    # The hypar objective: -log(1 + exp(-(f - c))) where c is a cost threshold
    # We approximate EHI using the GP posterior
    
    # Improvement over best (for minimization, lower is better)
    improvement = f_best - mu
    
    # Gaussian damping factor: suppresses scores where uncertainty is too large
    # This prevents over-exploration of high-uncertainty, high-cost regions
    # The damping scale is tied to the cost to balance exploration vs exploitation
    damping_scale = cost * 0.5
    damping = np.exp(-0.5 * (sigma / damping_scale) ** 2)
    
    # EHI approximation: use EI-like term but with hypar-inspired saturation
    # For the hypar objective, the expected improvement has a form that
    # saturates for large improvements (unlike linear EI)
    
    # Compute z-score
    z = improvement / sigma
    
    # Standard EI component
    ei_component = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei_component = np.maximum(ei_component, 0.0)
    
    # Hypar-inspired saturation: the benefit of improvement diminishes
    # for very large improvements (already well below f_best)
    # Use a soft saturation based on the improvement magnitude
    sat_scale = 0.1  # saturation scale in objective units
    saturation = 1.0 - np.exp(-ei_component / sat_scale)
    
    # Apply Gaussian damping
    damped_ehi = saturation * damping
    
    # Cost-aware: divide by cost to get per-unit-cost score
    # Use a slightly super-linear cost penalty to strongly discourage expensive evaluations
    cost_penalty = cost ** 1.5
    score = damped_ehi / cost_penalty
    
    # Safety: ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative scores
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
