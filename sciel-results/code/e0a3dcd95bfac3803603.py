"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as t_dist
from scipy.special import gamma

_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses t-Expected Improvement for robustness.
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: Pure exploitation (minimize mu)
        return -mu

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # t-Expected Improvement parameters
    nu = 5.0  # Degrees of freedom, 5 is a common robust choice
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate t-EI
    # T_nu(z) is the CDF of the t-distribution with nu degrees of freedom
    cdf_t = t_dist.cdf(z, df=nu)
    
    # PDF of t-distribution
    # pdf_t = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + z^2/nu)^(-(nu+1)/2)
    # We can use scipy.stats.t.pdf
    pdf_t = t_dist.pdf(z, df=nu)
    
    # EI_t = (f_best - mu) * T_nu(z) + sigma * pdf_t
    # Note: The standard EI formula is (f_best - mu) * Phi(z) + sigma * phi(z)
    # For t-distribution, the factor is slightly different if derived from scratch, 
    # but using the CDF and PDF directly in the same structure is a common approximation 
    # or exact if the variable is standardized t.
    # Actually, if Z ~ t_nu, E[max(0, f_best - (mu + sigma*Z))]
    # = (f_best - mu) P(Z < (f_best-mu)/sigma) + sigma E[Z * I(Z < z0)]
    # The term E[Z * I(Z < z0)] for t-dist is not just pdf(z0).
    # However, for large nu it approaches Gaussian.
    # Let's use the standard Gaussian EI but with a robust sigma if t-EI is too complex or risky.
    
    # Alternative: Use standard Gaussian EI but with a 'trust region' or just rely on its robustness.
    # Given the previous success of the mixed EI/UCB, let's try a cleaner EI.
    
    # Let's try Log-EI to prevent extreme values from dominating if sigma is noisy.
    # But first, let's try a standard EI with a small UCB component for early exploration.
    
    # Improvement
    improvement = f_best - mu
    
    # Gaussian EI
    z_g = improvement / sigma_safe
    cdf = t_dist.cdf(z_g, df=nu) # Using t-cdf for robustness
    pdf = t_dist.pdf(z_g, df=nu)
    
    # The exact t-EI formula involves a factor.
    # Let's stick to Gaussian EI for simplicity and reliability, but maybe add a small noise floor to sigma to prevent division by zero or extreme z.
    
    cdf_g = norm.cdf(z_g)
    pdf_g = norm.pdf(z_g)
    
    ei = improvement * cdf_g + sigma_safe * pdf_g
    
    # Add a small exploration term based on sigma to encourage visiting unexplored areas
    # if EI is low (i.e., we are not confident we can improve)
    # Score = EI + beta * sigma
    # Let's use a dynamic beta that starts high and decays.
    # We don't have step count, but we can use f_best as a proxy for progress.
    # If f_best is high, we are early. If low, we are late.
    
    # Beale function range on [-4.5, 4.5]^2 is roughly [0, 22516].
    # Initial f_best might be around 100-1000.
    # Optimal is 0.
    
    # Let's try a fixed blend that worked well or a pure EI.
    # The previous program used 0.7 EI + 0.3 UCB.
    # Let's try 1.0 EI. Pure EI is often best for smooth functions.
    
    score = ei
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# Need to import norm
from scipy.stats import norm

# EVOLVE-BLOCK-END
