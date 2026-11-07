"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Scaled Threshold Expected Improvement.
    
    The threshold for improvement is scaled by the current best value (f_best).
    T = f_best * (1 + lambda)
    
    This encourages exploration when f_best is large (early in optimization)
    by allowing points that are worse than f_best but better than T to have
    positive expected improvement. As f_best approaches the minimum (0),
    T approaches f_best, and the behavior converges to standard EI (exploitation).
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Scaling factor for the threshold.
    # A value of 1.0 means the threshold is 2 * f_best.
    # This provides significant early exploration.
    lambda_explore = 1.0
    
    # Ensure f_best is non-negative for the scaling logic to hold as intended
    # (though Weierstrass is non-negative, safety check)
    f_best_safe = max(f_best, 0.0)
    
    # Dynamic threshold
    threshold = f_best_safe * (1.0 + lambda_explore)
    
    # Calculate Expected Improvement relative to the dynamic threshold
    # EI(x) = E[max(threshold - f(x), 0)]
    z = (threshold - mu) / sigma_safe
    ei = (threshold - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    return ei
# EVOLVE-BLOCK-END
