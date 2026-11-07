import numpy as np
import time

# Global state for Thompson Sampling seed
# We need a consistent but evolving random state for the sampling.
# Since the function is called repeatedly in a loop, we can maintain a global seed
# or use a deterministic seed based on the current call context if possible.
# However, the signature is fixed. We can use a global counter to vary the sample
# slightly to avoid getting stuck in the same local optimum if the posterior is flat,
# or simply use a fixed seed per call if we want reproducibility.
# For Thompson Sampling, we typically draw a random sample from N(mu, sigma^2).
# To make it deterministic across runs for the same input (if required by evaluator),
# we might need to derive the random number from the inputs.
# But standard Thompson Sampling uses randomness.
# The evaluator runs seeds 0, 1, 2. The evaluator handles the initial random points.
# The acquisition function just needs to return scores.
# If we use np.random, it will be non-deterministic unless seeded.
# Let's use a global counter to ensure we don't pick the exact same sample every time
# if the posterior doesn't change much, or just use a fixed offset.
# Actually, for a single call, the scores are static.
# The "sampling" in TS is usually done by sampling the function value y ~ N(mu, sigma^2)
# and then picking the x that minimizes y (since we want to minimize).
# So score = -y_sample where y_sample ~ N(mu, sigma^2).
# To make it reproducible or at least stable, we can use a hash of the inputs to seed the random generator.
# Or, simpler: just use a fixed random state that evolves? No, state doesn't persist well in these tasks usually.
# Let's use a deterministic pseudo-random number based on the index and a global step counter?
# We don't have a step counter.
# Let's use a simple trick: sample from N(mu, sigma^2) using a standard normal variable Z.
# Z can be generated deterministically from the input mu/sigma? No, that's not TS.
# Let's assume the evaluator allows non-determinism or that we should use a fixed seed for the RNG inside the function.
# If we use a fixed seed, the "sample" is the same every time for the same mu/sigma.
# This is effectively a deterministic "sampled" UCB-like function.
# To add exploration, we can vary the seed based on the number of non-finite values or something?
# No, let's just use a global RNG that is initialized once.
# In Python, module-level variables persist.

_rng = np.random.default_rng(42)

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function.
    
    We simulate a draw from the posterior predictive distribution at each candidate point.
    Since we want to minimize the function, we want to pick the point with the lowest
    sampled value. The evaluator picks the candidate with the HIGHEST score.
    Therefore, we return score = -sampled_value.
    
    sampled_value ~ N(mu, sigma^2)
    
    To ensure we don't always pick the exact same point if the posterior is very certain,
    the randomness in Z handles this.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Generate a standard normal random variable for each candidate
    # Using the global RNG ensures different samples on each call (if the loop calls it multiple times)
    # Wait, the evaluator calls acquisition once per step.
    # If the RNG state persists, each step gets a different Z. This is good for exploration.
    Z = _rng.standard_normal(mu.shape)
    
    # Sampled posterior predictive value
    # y_sample = mu + sigma * Z
    # We want to minimize y_sample.
    # The evaluator maximizes the returned score.
    # So score = -y_sample
    scores = -(mu + sigma_safe * Z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
