"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use Probability of Improvement (PI) for minimization.
    
    PI calculates the probability that the function value at x is better 
    (lower) than the current best value f_best.
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    This approach is simpler than EI and tends to be more exploitative,
    which can be beneficial in later stages of optimization or for 
    functions with clear basins of attraction like the sphere function.
    """
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the normalized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute PI
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure all scores are finite
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END
