"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    This approach calculates the expected improvement over the current best value
    (f_best) for each candidate point. It balances exploitation (low mu) and
    exploration (high sigma) by integrating the probability of improvement
    weighted by the magnitude of the improvement.
    
    For minimization:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # Use a small epsilon
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Numerical stability: ensure scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
