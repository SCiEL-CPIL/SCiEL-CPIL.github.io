import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute Thompson Sampling scores for candidate points.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        Scores for each candidate, shape (N,), where higher is better.
    """
    # Generate a single random draw from the standard normal distribution
    # This z is shared across all points to create a correlated sample of the function surface
    z = np.random.randn()
    
    # Sample the function value: y = mu - sigma * z
    # Since we are minimizing, lower y is better.
    # The acquisition function should return higher scores for better candidates.
    # Therefore, we return the negative of the sampled value.
    scores = -(mu - sigma * z)
    
    return scores
