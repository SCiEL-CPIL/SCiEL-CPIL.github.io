"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB).
    
    UCB is effective for multimodal functions like Hartmann-3D because it explicitly
    balances exploitation (mu) and exploration (sigma). Unlike EI, which focuses on
    expected improvement over the current best, UCB can be tuned to explore more
    aggressively, which is helpful in the early stages of optimization with few samples.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Safety margin: if we're getting close to the 300s limit per eval, still return quickly
    # However, UCB calculation is O(n) and very fast, so this is rarely a bottleneck.
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB parameter. 
    # A higher beta encourages more exploration. 
    # For Hartmann-3D with only 25 optimization steps, aggressive exploration is beneficial.
    # Standard values range from 1.0 to 3.0. We choose 1.5 as a strong balance.
    beta = 1.5
    
    # Calculate UCB scores: mu + beta * sigma
    # Higher mu (lower function value, since we are minimizing? No, GP models the function values.
    # The task says "minimizes hartmann 3d". 
    # In standard BO for minimization:
    # If the GP models the raw function values f(x), then lower mu is better.
    # However, the contract says: "mu and sigma ... are in objective units".
    # "Scoring (lower is better): mean over the seeds of the simple regret, lowest observed value minus the minimum value."
    # "The evaluator measures the raw value (lower is better)."
    # This implies the objective function returns the raw Hartmann value.
    # Therefore, the GP is likely trained on these raw values.
    # We want to minimize f(x).
    # So, a low mu is good. A high sigma is good for exploration.
    # UCB for minimization is typically: -mu + beta * sigma  OR  -(mu - beta * sigma)
    # Let's verify the sign.
    # If we want to find the minimum, we want points with low expected value.
    # So we should select points with low mu.
    # Standard UCB for maximization: mu + beta * sigma.
    # Standard UCB for minimization: -mu + beta * sigma.
    
    # Let's re-read carefully.
    # "mu and sigma ... are in objective units."
    # "f_best is the lowest observed value."
    # If the GP models the function to be minimized, then mu is the predicted function value.
    # We want to pick x that minimizes f(x).
    # So we want low mu.
    # The score returned should be "higher meaning evaluate next".
    # So we need a score that is high for good candidates.
    # Good candidate = low mu, high sigma.
    # Score = -mu + beta * sigma.
    
    # Let's check the current program's logic.
    # Current: ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # If mu < f_best, z > 0, cdf(z) ~ 1, pdf(z) is small. EI is positive.
    # If mu > f_best, z < 0, cdf(z) ~ 0, EI is small/positive.
    # EI is always positive. Higher EI means more expected improvement.
    # This implies the GP is modeling the raw values, and we are minimizing.
    
    # So for UCB minimization:
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
