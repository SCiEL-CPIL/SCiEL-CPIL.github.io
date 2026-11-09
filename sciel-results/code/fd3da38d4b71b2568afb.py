"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    Higher score means more likely to be selected for evaluation.
    """
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, we want to maximize the probability that f(x) < f_best
    # Expected Improvement: EI = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x))
    
    # Standardize the improvement
    z = (f_best - mu) / sigma
    
    # Compute EI analytically
    from scipy.stats import norm
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Handle any potential NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
