"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses Thompson Sampling: sample from the posterior and score by negative sampled value.
    This is a randomized acquisition that naturally balances exploration and exploitation.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280.0

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)

    n = len(mu)
    if n == 0:
        return np.array([])

    # Check deadline
    if time.time() > deadline:
        # Fallback: pure LCB
        kappa = 1.0
        scores = -mu + kappa * sigma
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Thompson Sampling: sample a random function from the GP posterior
    # For each candidate, sample f ~ N(mu, sigma^2)
    # Score = -sampled_value (lower is better, so negate for "higher is better")
    
    # Use a fixed random state per call to ensure reproducibility within a run
    # But we want different samples each step, so use np.random directly
    rng = np.random.default_rng()
    
    # Sample from N(mu, sigma^2) for each candidate
    # Handle sigma == 0 cases: sampled value = mu
    samples = np.empty(n)
    mask = sigma > 1e-10
    
    if np.any(mask):
        samples[mask] = rng.normal(mu[mask], sigma[mask])
    
    if np.any(~mask):
        samples[~mask] = mu[~mask]

    # Score: higher is better, so we want to minimize the sampled value
    # score = -sampled_value
    scores = -samples

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
