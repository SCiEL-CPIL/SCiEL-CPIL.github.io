"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (N,) - posterior mean
    sigma: array of shape (N,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (N,) - higher score means evaluate next
    """
    start_time = time.time()
    
    # Ensure sigma is non-negative to handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw random samples from the standard normal distribution
    z = np.random.standard_normal(len(mu))
    
    # Compute samples from the posterior predictive distribution: y ~ N(mu, sigma^2)
    # y = mu + sigma * z
    samples = mu + sigma * z
    
    # Since we are minimizing the objective function, a lower sample 'y' is better.
    # The acquisition function must return higher values for better candidates.
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    # Deadline guard
    # The computation is O(N) and very fast, but we check just in case.
    if time.time() - start_time > 290:
        # If we are running out of time, we just return the computed scores.
        # No complex fallback needed as the main computation is already done.
        pass
        
    return scores
# EVOLVE-BLOCK-END
