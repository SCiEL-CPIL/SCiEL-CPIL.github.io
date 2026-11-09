"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Samples from the posterior GP distribution and returns a score
    based on the negative of the sampled value divided by the cost.
    Higher scores indicate better candidates to evaluate next.
    """
    # Ensure sigma is positive to avoid numerical issues, though GP should handle this
    sigma = np.maximum(sigma, 1e-9)
    
    # Generate a single sample from the standard normal distribution
    # Shape matches mu and sigma (2000,)
    z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # The directive suggests mu - sigma * z. Since z is symmetric, this is equivalent to mu + sigma * z.
    y = mu - sigma * z
    
    # We want to minimize y / cost.
    # The acquisition function must return scores where HIGHER is better.
    # Therefore, we return the negative of the quantity we want to minimize.
    scores = -y / cost
    
    # Ensure finite scores
    # If any non-finite values appear (e.g., due to extreme outliers in sampling, though unlikely with normal dist),
    # replace them with a very negative value so they are not selected.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
