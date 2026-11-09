"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0  # Seconds, with safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 10:
        # Fallback: simple cost-aware score to avoid timeout
        # Ensure finite and non-negative
        c = np.maximum(cost, 1e-9)
        return -np.log1p(c)

    # Ensure inputs are numpy arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any non-finite values in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: Sample from the posterior N(mu, sigma^2)
    # We want to minimize, so we sample a value y ~ N(mu, sigma^2)
    # The acquisition score should favor low y (good values) and low cost.
    
    # Standard normal random numbers
    # Using a fixed seed for reproducibility in the acquisition step? 
    # No, Thompson Sampling requires randomness per call to explore.
    # However, the evaluator calls acquisition once per step with 2000 candidates.
    # We generate random samples for the 2000 candidates.
    
    # To ensure we don't use too much time or memory, we generate randoms efficiently
    # Note: np.random.normal is fast for 2000 elements.
    z = np.random.normal(0, 1, size=mu.shape)
    samples = mu + sigma_safe * z
    
    # We want to maximize a score.
    # Since we are minimizing the function, lower samples are better.
    # So, score = -samples.
    
    # Cost-awareness:
    # The cost is 1 + mean(x scaled to [0, 1]).
    # We want to penalize high cost.
    # A common approach is to divide by cost, but if cost is small, this can be noisy.
    # Alternatively, use log-cost or a soft penalty.
    # Let's try: score = -samples / cost_safe
    # This is similar to EI/cost but with a random sample.
    
    # However, simple division can lead to very high scores for very low cost regions
    # even if the sample is not great.
    # Let's normalize the samples first?
    # The objective is to minimize.
    # f_best is the current best.
    # We can shift the samples by f_best to make them comparable?
    # Or just use -samples.
    
    # Let's look at the scale.
    # mu and sigma are in objective units.
    # f_best is the lowest observed value.
    # If we use -samples, we are directly picking the point with the lowest sampled value.
    # This is pure exploitation of the current posterior sample.
    # To balance exploration, the randomness in `z` handles it.
    
    # Cost adjustment:
    # Dividing by cost favors cheap points.
    # Let's try: score = -samples / cost_safe
    
    # To prevent extreme values, we can clip the scores.
    # Or normalize them.
    # Let's try a simple normalization:
    # score = (-samples - min(-samples)) / (max(-samples) - min(-samples) + 1e-9)
    # This maps scores to [0, 1].
    # But we need to incorporate cost.
    
    # Alternative:
    # score = -samples * (1 / cost_safe)
    # This is the same as -samples / cost_safe.
    
    # Let's check if this is stable.
    # If sigma is very small, samples ~ mu.
    # If mu is very low (good), score is high (good).
    # If cost is low, score is higher (good).
    
    # One issue: Thompson Sampling can sometimes pick a very bad point if the noise is high.
    # But over many iterations, it should find the minimum.
    
    # Let's add a small exploration bonus to points with high uncertainty?
    # No, Thompson Sampling already does this. High sigma -> wider spread of samples -> chance of picking a very low (good) sample.
    
    # Let's refine the cost term.
    # Cost is in [1, 2] roughly (1 + mean(x) where x in [0,1]).
    # So 1/cost is in [0.5, 1].
    # This is a mild penalty.
    
    # Let's try:
    # score = -samples / cost_safe
    
    # Ensure finite
    scores = -samples / cost_safe
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If all scores are very similar, or if we want to ensure some spread,
    # we might need to normalize.
    # But the evaluator just picks the argmax.
    # So relative scale doesn't matter, only ordering.
    
    # However, if the GP is not well-calibrated, mu and sigma might be off.
    # Let's add a small epsilon to cost to avoid division by zero (already done).
    
    # One potential problem: if sigma is 0 everywhere (converged), samples = mu.
    # Then we pick the point with lowest mu / cost.
    # This is good.
    
    # Let's return the scores.
    return scores
# EVOLVE-BLOCK-END
