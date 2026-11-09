"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (N,) posterior mean
    sigma: array of shape (N,) posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (N,) scores where higher is better.
    """
    # Thompson Sampling: Sample from the posterior predictive distribution
    # f(x) ~ N(mu(x), sigma(x)^2)
    
    # Generate standard normal random variables
    # We use a fixed seed per call? No, the evaluator handles the loop.
    # The randomness comes from np.random.
    
    eps = np.random.standard_normal(len(mu))
    
    # Sample from the posterior
    # sample = mu + sigma * eps
    # Since we are minimizing the objective, we want to select the candidate
    # with the lowest sampled value.
    # The contract says "higher meaning evaluate next".
    # So we return the negative of the sample.
    sample = mu + sigma * eps
    scores = -sample
    
    return scores
# EVOLVE-BLOCK-END
