"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a sample from the GP posterior and scores candidates by the 
    expected improvement over the best observed value, divided by the 
    evaluation cost. This prioritizes points that are likely to improve 
    the best value while being cheap to evaluate.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : array_like, shape (2000,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from the posterior: y ~ N(mu, sigma^2)
    noise = np.random.normal(0, 1, size=mu.shape)
    samples = mu + sigma_safe * noise
    
    # Calculate cost-adjusted improvement
    # We want to maximize (f_best - sample) / cost
    # This rewards samples that are lower than f_best and penalizes high costs
    scores = (f_best - samples) / cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
