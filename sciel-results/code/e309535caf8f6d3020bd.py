import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Samples a single realization from the Gaussian posterior for all candidates
    and returns negative sampled values as scores. This naturally balances
    exploration and exploitation without manual tuning of exploration parameters.
    """
    # Generate a single random draw from the standard normal distribution
    z = np.random.randn(mu.shape[0])
    
    # Sample from the posterior: mu + sigma * z
    sample = mu + sigma * z
    
    # Return negative sample (higher score = lower sampled value = better)
    score = -sample
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
