"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_ACQ_START_TIME = time.time()

# Use a fixed seed for reproducibility within a single run if needed, 
# but typically we want random sampling. However, to ensure the function is 
# deterministic given the same inputs (which might be required for some evaluators 
# or to avoid flakiness in scoring comparisons), we should be careful.
# The task says "Return 2000 finite scores". It doesn't explicitly forbid randomness.
# Thompson Sampling is inherently stochastic.
# We will generate the random normal variables inside the function.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling.
    Samples from the posterior and scores based on sampled improvement per unit cost.
    """
    # Deadline guard
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 290:
        # Fallback to a simple greedy-like score if time is running out
        # Use mean only, no sampling
        scores = (f_best - mu) / (cost + 1e-12)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Generate random normal samples for Thompson Sampling
    # Shape of mu and sigma is (2000,)
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled values from the posterior
    # sigma is standard deviation
    sampled_values = mu + sigma * z
    
    # Calculate sampled improvement: f_best - sampled_value
    # If sampled_value < f_best, improvement is positive.
    # We divide by cost to make it cost-aware.
    cost_safe = cost + 1e-12
    scores = (f_best - sampled_values) / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
