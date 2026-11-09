# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.special import erfinv

# Module-level state for time tracking
_start_time = None
_DEADLINE_MARGIN = 15.0  # seconds before the 330s limit

def _get_elapsed():
    """Return elapsed time since module load, initializing on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded():
    """Check if we are close to the 330s wall-clock limit."""
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Student-t Expected Improvement (EI-t) acquisition function.
    
    Uses a t-distribution with nu=3 degrees of freedom to model the
    posterior, which has heavier tails than the Gaussian. This encourages
    exploration of distant uncertain regions while still exploiting known
    good areas.
    
    Score to maximize: EI_t(x) = E[max(f_best - Y, 0)] where Y ~ t_nu(mu, sigma)
    
    For minimization, higher score means more promising to evaluate next.
    """
    # Ensure sigma is positive and finite
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Deadline guard: if we're close to the limit, use a fast fallback (standard EI)
    if _is_deadline_exceeded():
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
        if not np.any(np.isfinite(ei_scores)):
            return np.zeros_like(mu)
        return ei_scores
    
    # Student-t EI formula
    # For a t-distribution with nu degrees of freedom, location mu, scale sigma:
    # EI = sigma * [ (z * t_pdf_nu(z) + (1 + z^2/nu) * (1 - t_cdf_nu(z)) ) ]
    # where z = (f_best - mu) / sigma
    #
    # For nu=3:
    # t_pdf_3(z) = (1+2/3) * (1+z^2/3)^(-2) * (1/sqrt(3*pi)) ... 
    # Actually, let's use the general formula.
    
    # The t-distribution with nu DOF, location loc, scale scale:
    # pdf(x) = (1+((x-loc)/scale)^2/nu)^(-(nu+1)/2) * Gamma((nu+1)/2) / (sqrt(nu*pi)*scale*Gamma(nu/2))
    # cdf(x) = 0.5 * (1 + sign(z) * I_{nu/(nu+z^2)}(nu/2, 1/2)) where z=(x-loc)/scale
    #          = 1 - 0.5 * (1 - sign(z) * I_{nu/(nu+z^2)}(nu/2, 1/2)) for z>0
    
    # For minimization EI:
    # Let Z = (f_best - mu) / sigma
    # EI = sigma * [ Z * pdf_t(Z) + (1 + Z^2/nu) * (1 - CDF_t(Z)) ]
    #     where pdf_t and CDF_t are for standard t with nu DOF
    
    nu = 3.0
    Z = (f_best - mu) / sigma_safe
    
    # Compute t-distribution pdf and cdf for standard t (loc=0, scale=1)
    # pdf_t(z) = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * (1+z^2/nu)^(-(nu+1)/2)
    # For nu=3:
    #   Gamma(2) = 1, Gamma(1.5) = sqrt(pi)/2
    #   pdf_t(z) = 1 / (sqrt(3*pi) * sqrt(pi)/2) * (1+z^2/3)^(-2)
    #            = 2 / (pi * sqrt(3)) * (1+z^2/3)^(-2)
    
    # General computation using scipy
    from scipy.stats import t as t_dist
    
    # Compute pdf and cdf of standard t distribution at Z
    # t_dist.pdf(z, df=nu) and t_dist.cdf(z, df=nu)
    t_pdf = t_dist.pdf(Z, df=nu)
    t_cdf = t_dist.cdf(Z, df=nu)
    
    # EI_t = sigma * [ Z * t_pdf(Z) + (1 + Z^2/nu) * (1 - t_cdf(Z)) ]
    ei_scores = sigma_safe * (Z * t_pdf + (1.0 + Z**2 / nu) * (1.0 - t_cdf))
    
    # Ensure all scores are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
    
    # If all are -inf, return zeros
    if not np.any(np.isfinite(ei_scores)):
        return np.zeros_like(mu)
    
    return ei_scores
# EVOLVE-BLOCK-END
