"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses pure Expected Improvement (EI) for minimization.
    This strategy focuses entirely on exploiting the GP's posterior
    to find points likely to improve upon the current best value,
    avoiding the potential conflicts of hybrid exploration strategies
    which can dilute the signal in low-dimensional, well-behaved problems.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Ensure sigma is positive to avoid division by zero
    # Use a very small epsilon to prevent numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # We want to minimize f, so improvement is (f_best - f_new)
    # E[I] = E[max(0, f_best - f_new)]
    # If f_new ~ N(mu, sigma^2), then:
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # Using scipy.stats.norm for numerical stability
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    # Due to floating point errors, ei might be slightly negative
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
