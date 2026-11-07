"""Rewritten program using Thompson Sampling for cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores using Thompson Sampling.
    
    Args:
        mu: Array of mean predictions, shape (N,)
        sigma: Array of standard deviations, shape (N,)
        f_best: Best observed value (scalar)
        cost: Array of costs, shape (N,)
        
    Returns:
        Array of scores, shape (N,), where higher is better.
    """
    # Ensure sigma is positive to avoid division or sampling issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # We draw one sample per candidate
    z = np.random.randn(len(mu))
    y_samples = mu + sigma * z
    
    # Calculate the improvement over the current best
    # Since we are minimizing, improvement is f_best - y_sample
    improvements = f_best - y_samples
    
    # Cost-aware score: Improvement per unit cost
    # Higher score indicates a better candidate
    scores = improvements / cost
    
    # Handle any non-finite values (e.g., if cost was 0, though unlikely)
    # Replace non-finite scores with a very low value so they are not selected
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
