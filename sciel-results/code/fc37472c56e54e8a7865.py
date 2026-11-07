"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG estimates the expected reduction in the global minimum value.
    For a candidate x, if we observe y ~ N(mu(x), sigma(x)), the expected
    improvement in f_best is E[max(0, f_best - y)].
    
    However, the full KG would also account for how the observation updates
    the GP posterior for all other points. A common approximation is to use
    the Expected Improvement (EI) of the candidate itself, which is what we
    compute here. This is a well-known surrogate for KG that is much cheaper
    to compute and often performs comparably.
    
    For minimization:
    EI(x) = E[max(0, f_best - f(x))]
         = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    This is equivalent to the standard EI formula for minimization.
    """
    # Handle edge cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # EI is non-negative by construction, but clip just in case
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
