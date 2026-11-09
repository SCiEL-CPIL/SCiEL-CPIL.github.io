"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Pre-compute a fixed set of random numbers for Thompson Sampling.
# We use a deterministic seed so that the behavior is reproducible across calls,
# but we vary the "sample index" based on a hash of f_best to avoid
# always picking the same point if the posterior hasn't changed much.
# However, since we don't have access to the step count, we rely on the
# stochastic nature of the samples. To ensure diversity, we pre-compute
# a large batch of standard normal samples.

# The Weierstrass function is known to have many local minima.
# Thompson Sampling (TS) is often superior to EI for multimodal functions
# because it samples from the posterior, allowing it to explore different
# modes more effectively.

# We will use a hybrid approach:
# 1. Primary: Thompson Sampling. We sample one value from the posterior
#    at each candidate point and select the candidate with the lowest sample.
#    To make this robust, we can average over a few samples or use a
#    single sample with a fixed RNG seed that changes based on f_best.
#    Actually, a common effective variant is to sample the function value
#    and pick the argmin.

# Let's implement a robust Thompson Sampling:
# score_i = mu_i + sigma_i * z_i
# where z_i is a standard normal random variable.
# We want to maximize the score (since higher score means evaluate next).
# Wait, the function minimization means we want low values.
# The acquisition function returns scores where HIGHER is better.
# So for TS, we want to pick the point with the LOWEST predicted value.
# Thus, the "score" for the acquisition function should be something like:
# - (mu_i + sigma_i * z_i)
# This way, the point with the lowest sampled value gets the highest score.

# To ensure we don't always pick the same point due to RNG correlation,
# we can use a deterministic pseudo-random sequence based on the index
# and a global counter or f_best.
# Since we don't have a global counter, we can use a hash of f_best
# to initialize the RNG for this specific call. This ensures that
# as f_best improves, the "randomness" changes, encouraging exploration
# in different regions.

# However, calling np.random.seed inside the function is bad practice
# if the function is called in parallel or if state is shared.
# But the task says "No coordinates are passed" and implies a single thread.
# Let's use a local RandomState seeded by a hash of f_best and the step
# (which we don't have).
# A simpler, robust approach: Use a fixed large array of random numbers
# and index into it based on a hash of f_best.

# Let's try a different structural change:
# Use a combination of EI and TS, or just a better EI variant like UCB.
# UCB: score = -mu + kappa * sigma
# This is deterministic and fast.
# But TS is measured to be better (5.141 vs 1.0).

# Let's implement Thompson Sampling with a deterministic but varying seed.
# We will generate a vector of z-scores.
# To avoid the same z-vector every time (which would lead to the same pick
# if mu/sigma are identical), we vary the seed based on f_best.

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling:
    # We sample from the posterior: y_i ~ N(mu_i, sigma_i^2)
    # We want to minimize y_i.
    # So we assign a score of -y_i.
    # score_i = -(mu_i + sigma_i * z_i)
    
    # To get diverse samples, we use a RandomState seeded by a hash of f_best.
    # This ensures that as the best value changes, the random draws change,
    # preventing the algorithm from getting stuck in a local mode due to
    # fixed random offsets.
    
    # Hash f_best to get an integer seed
    # f_best is a float. We can use its bits or a simple hash.
    # Using a simple hash function on the bytes of the float.
    seed = int(np.abs(np.sin(f_best) * 100000)) % (2**32 - 1)
    
    rng = np.random.RandomState(seed)
    
    # Generate standard normal samples
    z = rng.standard_normal(len(mu))
    
    # Sampled values
    sampled_values = mu + sigma * z
    
    # Score is negative of sampled value (since we want to minimize)
    scores = -sampled_values
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
