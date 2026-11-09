"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 in 8D.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - lowest observed value
    - cost: array of shape (2000,) - cost of evaluating each point (1 + mean(x_scaled))
    
    Returns:
    - score: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    t_start = time.time()
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: sample from the posterior of the improvement
    # For each candidate, improvement ~ N(f_best - mu, sigma^2)
    # We want high improvement (low function value), so we sample the function value
    # and use negative value as score (higher = better)
    
    # Sample from the posterior: x_sample = mu + sigma * z, z ~ N(0,1)
    # Use a fixed seed for reproducibility within a call? No, we want stochasticity.
    # But we need to return deterministic scores for the same inputs?
    # The task says "Return 2000 finite scores". It doesn't explicitly say deterministic.
    # However, standard BO loops usually expect deterministic acquisition or handle randomness internally.
    # If I use randomness, the score changes every time acquisition is called with same inputs.
    # The evaluator calls acquisition once per iteration with 2000 candidates.
    # So stochasticity is acceptable and intended for TS.
    
    # Generate standard normal random numbers
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    sampled_values = mu + sigma_safe * z
    
    # We want to minimize, so we prefer low sampled values.
    # Score = -sampled_values (higher is better)
    # But we also need to be cost-aware.
    # Apply cost penalty to the score.
    
    # Cost-aware Thompson Sampling:
    # score = -sampled_values * gate
    # Or: score = (-sampled_values - lambda * cost)
    
    # Let's stick to the multiplicative gate design family but with TS scores.
    # The previous design used EI * exp(-k * cost).
    # Here we use (-sampled_value) * exp(-k * cost).
    
    # Wait, sampled_value can be positive or negative.
    # If sampled_value is very negative (good), -sampled_value is large positive.
    # If sampled_value is positive (bad), -sampled_value is negative.
    # Multiplying by a positive gate preserves the ordering.
    
    # What k to use? The previous best inspiration used k=1.0 for exponential penalty.
    # Current program uses k=0.3.
    # Let's try k=0.5 as a middle ground, or stick to the successful k=1.0?
    # The lesson says "Exponential cost penalty (lambda=1) ... Measured: 1 -> 1.04".
    # So k=1.0 is a strong candidate.
    
    k = 1.0
    gate = np.exp(-k * cost)
    
    # Base score from Thompson Sampling
    # We want to maximize the probability of finding a point better than f_best.
    # In TS, we sample the function value and pick the one with the lowest value.
    # So score = -sampled_value.
    # To make it cost-aware, we penalize high cost.
    
    # One common way: score = -sampled_value * gate
    # Another: score = -sampled_value - penalty * cost
    
    # Let's use the multiplicative gate to stay close to the existing design family.
    score = -sampled_values * gate
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
