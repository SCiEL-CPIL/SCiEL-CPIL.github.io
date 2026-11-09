"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Returns scores where higher is better (i.e., we return -sampled_value).
    """
    # Simple deadline guard, though vectorized ops are very fast
    # The limit is 300s per evaluation, but this function is called many times.
    # The total budget is 330s for the whole run.
    # We assume the caller handles the global budget, but we ensure this function
    # doesn't hang.
    
    # Generate random standard normal variables
    z = np.random.randn(len(mu))
    
    # Sample from the posterior predictive distribution
    # y = mu + sigma * z
    # Since we are minimizing, we want to find the minimum y.
    # The interface asks for scores where HIGHER is better.
    # So we return -y.
    samples = mu - sigma * z
    scores = -samples
    
    return scores
# EVOLVE-BLOCK-END
