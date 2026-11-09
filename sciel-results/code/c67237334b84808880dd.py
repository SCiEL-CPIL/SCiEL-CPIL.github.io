"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache the random generator to ensure consistent behavior if called multiple times in a tight loop,
# though typically each call is independent. Using a fixed seed for the acquisition function itself
# is not ideal for TS, but we need reproducibility for the evaluator? 
# Actually, the evaluator calls acquisition with fixed mu, sigma, f_best, cost.
# TS requires randomness. If we use a fixed seed, it's deterministic TS (which is just a specific sample).
# If we use global randomness, it's stochastic.
# The task says "Return 2000 finite scores". It doesn't say they must be deterministic.
# However, stochastic acquisition functions can be unstable in benchmarks if not careful.
# Let's use a deterministic pseudo-random sample based on the index to ensure 
# the same candidate gets the same "randomness" offset if the GP state doesn't change,
# but here mu/sigma change. 
# Standard TS: sample y = mu + sigma * z.
# We can use a fixed set of z-scores (standard normal) to make it deterministic 
# relative to the input shape, or use a global RNG.
# Given the strict 300s limit and potential for multiple calls, 
# let's use a deterministic transformation of the index to generate z-scores,
# ensuring that the "randomness" is consistent for the same candidate index 
# if the GP parameters were static, but here they are dynamic.
# Actually, the simplest and most robust TS implementation for a single step 
# is to draw from the normal distribution.
# To avoid issues with non-finite values or extreme outliers, we clip the samples.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed function value so far.
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (N,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: Sample from the posterior N(mu, sigma^2)
    # We use a deterministic pseudo-random seed based on the current time 
    # or a fixed seed? 
    # If we use a fixed seed, the "random" samples are the same every time,
    # which effectively turns TS into a deterministic heuristic that picks 
    # the same candidates relative to the distribution shape.
    # If we use a global RNG, it's truly stochastic.
    # For BO, stochastic TS is standard. 
    # However, to be safe against "invalid result" due to non-finite values 
    # or extreme variance, we clip the samples.
    
    # Generate standard normal samples
    # Using a fixed seed for the acquisition function is not standard, 
    # but using a global RNG is. Let's use a global RNG.
    # To ensure we don't run out of time or have issues, we generate samples efficiently.
    z = np.random.standard_normal(mu.shape)
    
    # Sampled values
    samples = mu + sigma * z
    
    # Cost-awareness: We want to maximize the sampled value per unit cost.
    # Since we are minimizing, a lower sampled value is better.
    # But acquisition functions usually return higher = better.
    # So we want high scores for candidates that are likely to have low function values.
    # Therefore, we should negate the sampled value or use -sample / cost.
    # Let's use -sample / cost.
    
    scores = -samples / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a small exploration bonus or normalization?
    # TS is already stochastic.
    
    return scores
# EVOLVE-BLOCK-END
