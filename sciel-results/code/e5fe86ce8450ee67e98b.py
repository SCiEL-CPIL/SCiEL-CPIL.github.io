"""Rewritten program using Thompson Sampling for Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of mean predictions (shape: (2000,))
        sigma: Array of standard deviation predictions (shape: (2000,))
        f_best: Current best observed value (lowest)
        
    Returns:
        scores: Array of scores (shape: (2000,)). 
                Higher scores indicate better candidates to evaluate.
                Since we are minimizing, we return negative sampled values.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior distribution for each candidate
    # y_i ~ N(mu_i, sigma_i)
    # We use a single random draw for all candidates.
    # Note: The evaluator calls this function. The randomness is internal.
    # To make it robust, we rely on numpy's global random state or 
    # we can use a fixed seed if we want determinism, but usually 
    # BO acquisition functions are stateless or use global state.
    # Given the context, using np.random is standard.
    
    z = np.random.randn(mu.shape[0])
    samples = mu + sigma * z
    
    # We want to minimize the function.
    # Thompson Sampling selects the point with the lowest sampled value.
    # The interface requires returning scores where HIGHER is better.
    # So we return the negative of the sampled values.
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
