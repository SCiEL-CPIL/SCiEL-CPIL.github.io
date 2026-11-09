"""Rewritten program using Thompson Sampling for Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of computing a deterministic score like Expected Improvement or UCB,
    we sample a function value from the posterior distribution for each candidate.
    We then select the candidate with the lowest sampled value (by returning negative
    scores, so the evaluator picks the max).
    
    This approach naturally balances exploration (high sigma) and exploitation (low mu)
    without requiring a manual tuning parameter like beta in UCB or xi in PI.
    """
    # Ensure sigma is strictly positive to avoid numerical issues in sampling
    # GP sigma should be non-negative, but clip to be safe
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw random samples from the standard normal distribution
    # Shape matches mu (2000,)
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the Gaussian Process posterior: y ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # We want to MINIMIZE the function.
    # The evaluator selects the point with the HIGHEST acquisition score.
    # Therefore, we invert the sampled values: lower function value -> higher score.
    scores = -samples
    
    # Ensure all returned scores are finite numbers
    # Replace any potential NaNs or Infs with large negative/positive values
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
