# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) acquisition function.
    # mu: array of predicted means, shape (n,)
    # sigma: array of predicted standard deviations, shape (n,)
    # f_best: best observed value so far (lower is better for minimization)
    
    # Guard against division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score: (f_best - mu) / sigma
    # For minimization, improvement is when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
