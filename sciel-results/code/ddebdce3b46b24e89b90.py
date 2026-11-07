"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    This approach implements a stochastic hypothesis testing strategy. 
    Instead of computing a deterministic score like Expected Improvement, 
    we draw a sample from the posterior distribution of the objective function 
    at each candidate point. The candidate with the lowest sampled value 
    (which corresponds to the highest score in a maximization context) is selected.
    
    This naturally balances exploration and exploitation: points with high 
    uncertainty (sigma) have a wider distribution of possible values, 
    increasing the chance that a sample will be very low (good), while 
    points with low uncertainty but good mean (mu) will consistently 
    produce good samples.
    
    Parameters:
    mu: np.ndarray, shape (2000,), mean of the GP posterior
    sigma: np.ndarray, shape (2000,), std dev of the GP posterior
    f_best: float, current best observed value (minimum)
    
    Returns:
    scores: np.ndarray, shape (2000,), higher scores indicate higher priority
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random variables for each candidate
    # This represents the stochastic component of the Thompson Sampling
    z = np.random.normal(0.0, 1.0, size=mu.shape)
    
    # Compute the posterior sample y ~ N(mu, sigma^2)
    y = mu + sigma_safe * z
    
    # For minimization, we want the lowest y.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    scores = -y
    
    # Handle any potential numerical non-finiteness (though unlikely with normal dist)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
