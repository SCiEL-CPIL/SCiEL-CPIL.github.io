"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as student_t


def acquisition(mu, sigma, f_best):
    """
    Student-t Expected Improvement acquisition function.
    
    Uses the t-distribution CDF/PDF with moderate degrees of freedom (nu=4)
    to get heavier tails than the standard Gaussian EI, encouraging exploration
    in uncertain regions while still exploiting areas with low predicted mean.
    
    Parameters:
        mu: array of shape (N,), posterior mean in objective units
        sigma: array of shape (N,), posterior standard deviation in objective units
        f_best: scalar, lowest observed value so far
    
    Returns:
        scores: array of shape (N,), higher means more promising to evaluate next
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Degrees of freedom for the Student-t distribution
    # Lower nu = heavier tails = more exploration
    # nu=4 gives a good balance between exploration and exploitation
    nu = 4.0
    
    # Student-t CDF and PDF
    t_cdf = student_t.cdf(z, df=nu)
    t_pdf = student_t.pdf(z, df=nu)
    
    # Expected improvement with Student-t
    # E[I] = (f_best - mu) * P(X < f_best) + sigma * pdf_t(z)
    # where the probability and density are from the t-distribution
    scores = (f_best - mu) * t_cdf + sigma_safe * t_pdf
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
