"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as student_t


def acquisition(mu, sigma, f_best):
    """
    Student-t Expected Improvement acquisition function.
    
    Uses a Student-t distribution with 3 degrees of freedom instead of
    Gaussian, providing heavier tails for better exploration of the
    sharp Beale function landscape.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Student-t with 3 degrees of freedom
    nu = 3.0
    
    # CDF and PDF of Student-t
    t_cdf = student_t.cdf(z, nu)
    t_pdf = student_t.pdf(z, nu)
    
    # Student-t Expected Improvement:
    # EI = (f_best - mu) * P(X < f_best) + sigma * E[(f_best - X)^+ | X < f_best] * P(X < f_best)
    # For Student-t: E[(f_best - X)^+ | X < f_best] involves the t-distribution
    # Simplified form: EI = (f_best - mu) * t_cdf(z) + sigma * t_pdf(z) * (nu / (nu - 1))
    # More precisely, for Student-t with nu degrees of freedom:
    # EI = (f_best - mu) * t_cdf(z) + sigma * t_pdf(z) * (1 + z * t_cdf(z) / t_pdf(z)) / (nu - 1) * nu
    # But a common approximation that works well:
    
    # The exact Student-t EI formula:
    # EI_t = (f_best - mu) * F_t(z) + sigma * f_t(z) * (nu / (nu - 1)) * (1 - z * f_t(z) / F_t(z))
    # where F_t and f_t are CDF and PDF of Student-t
    
    # A simpler and more robust form:
    # EI = (f_best - mu) * t_cdf(z) + sigma * t_pdf(z) * (nu / (nu - 1))
    
    # Use the standard EI form with Student-t:
    # The expected improvement under Student-t assumption
    # EI = (f_best - mu) * Phi_t(z) + sigma * phi_t(z) * (nu / (nu - 1))
    
    # Handle edge cases where t_cdf or t_pdf might be 0
    mask = t_pdf > 1e-15
    
    ei = np.zeros_like(mu)
    ei[mask] = (f_best - mu[mask]) * t_cdf[mask] + sigma_safe[mask] * t_pdf[mask] * (nu / (nu - 1))
    
    # For points where t_pdf is essentially 0, use the Gaussian approximation as fallback
    ei[~mask] = (f_best - mu[~mask]) * norm.cdf(z[~mask]) + sigma_safe[~mask] * norm.pdf(z[~mask])
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
