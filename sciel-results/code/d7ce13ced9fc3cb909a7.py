import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for Bayesian optimization.
    """
    # Ensure sigma is not zero to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement acquisition function
    # EI favors points where we expect improvement over f_best
    improvement = f_best - mu  # Higher is better
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    return ei
