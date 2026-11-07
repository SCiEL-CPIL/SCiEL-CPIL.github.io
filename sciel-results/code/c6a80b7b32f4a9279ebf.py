"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Thompson Sampling (TS)
    # Instead of maximizing a deterministic acquisition value like EI,
    # we sample a single value from the posterior distribution for each candidate.
    # This naturally balances exploration and exploitation:
    # - Candidates with high mu (low function value) are likely to be sampled low (good).
    # - Candidates with high sigma have a higher variance, so they have a higher chance
    #   of producing an extremely low sample (exploration).
    
    # Generate random normal noise for the 2000 candidates
    # Using a fixed seed per call is not possible as we don't have state, 
    # but we can use global RNG state which is fine for stochastic acquisition.
    # To ensure reproducibility in a specific run context if needed, we rely on the default RNG.
    # However, for a single acquisition call, we just draw one sample.
    
    # Sample from N(mu, sigma^2)
    # We want to minimize f, so we want samples to be as low as possible.
    # The acquisition score should be "higher = evaluate next".
    # So we return -sample.
    
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # 2. Cost-aware Scaling
    # The cost is 1 + mean(x_scaled).
    # We want to penalize high cost. 
    # A common approach is to divide by cost, or use a power law.
    # Given the cost is in [1, 2] (since x_scaled in [0,1], mean in [0,1]),
    # dividing by cost is a strong penalty.
    # Let's try dividing by sqrt(cost) to be less aggressive, or cost^0.5.
    # Actually, let's look at the structure: cost = 1 + mean(x).
    # If we divide the "improvement" by cost, we get improvement per unit cost.
    
    # However, TS samples are in objective units.
    # If f_best is very low (close to min), the samples might be negative or positive.
    # We want to select the point that likely has the lowest f.
    
    # Let's refine the cost adjustment.
    # If we simply do -sample / cost, we are optimizing for "lowest sample per cost".
    # This might be too aggressive.
    
    # Alternative: Cost-Adjusted Thompson Sampling.
    # Score = -sample / cost^alpha
    # Let's use alpha = 1.0 for simplicity first, or 0.5.
    # Let's try alpha = 0.5 to soften the cost penalty slightly, 
    # allowing expensive points if they have very promising samples.
    
    cost_safe = np.maximum(cost, 1e-9)
    alpha = 0.5
    cost_weight = cost_safe ** alpha
    
    score = -samples / cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
