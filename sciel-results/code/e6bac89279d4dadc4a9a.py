"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - Thompson Sampling scores (higher = better)
    """
    # Time guard: ensure we finish within reasonable time
    start_time = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw samples from posterior N(mu, sigma)
    # We want to minimize, so we sample from N(mu, sigma) and prefer lower values
    # But we need to be cost-aware, so we normalize by cost
    
    # Draw a single sample for each candidate
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Cost-aware scoring: we want to minimize samples / cost
    # Higher score = better, so we use negative of samples/cost
    # But we also want to ensure exploration in uncertain regions
    
    # Use a combination: primarily Thompson Sampling with a small exploration bonus
    # The exploration bonus rewards high uncertainty relative to the improvement potential
    
    # Base score: negative of sampled value divided by cost (lower sampled value + lower cost = better)
    base_score = -samples / np.maximum(cost, 1e-8)
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps when the sampled value is close to f_best but uncertainty is high
    uncertainty_bonus = sigma_safe / np.maximum(cost, 1e-8)
    
    # Combine: weight the Thompson sample primarily, with a small exploration component
    # The weight for exploration should be small to not override the sampling
    exploration_weight = 0.1
    
    scores = base_score + exploration_weight * uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check: if we've spent too much time, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
