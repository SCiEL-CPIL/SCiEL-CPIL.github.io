"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for minimizing Hartmann 6d.
    
    Samples a single objective value from the posterior at each candidate
    and returns the negative of that sample as the score (higher score =
    lower sampled value = preferred). This naturally balances exploration
    and exploitation without tuning hyperparameters.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from the posterior at each point
    # For minimization, we want the lowest sampled value
    # Use a fixed random state per call? No, we want randomness each call.
    # The randomness is the key to exploration.
    samples = mu - sigma_safe * np.random.randn(len(mu))
    
    # Score: higher is better. Since we minimize, lower sample is better.
    # So score = -sample
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
