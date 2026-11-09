"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as t_dist


def acquisition(mu, sigma, f_best):
    """
    Student-t Expected Improvement acquisition function.
    
    Uses the t-distribution (with moderate degrees of freedom) instead of
    the normal distribution to give heavier tails, encouraging exploration
    of distant uncertain regions in the multi-modal Weierstrass landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Degrees of freedom for the t-distribution.
    # Lower df -> heavier tails -> more exploration.
    # We use df=3 for a moderately heavy-tailed distribution.
    df = 3.0
    
    # Improvement: how much this point could beat f_best
    # Since we're minimizing, improvement = f_best - mu
    improvement = f_best - mu
    
    # Standardize using the t-distribution
    # z = improvement / sigma, but we use t-distribution CDF/PDF
    z = improvement / sigma_safe
    
    # For the t-distribution:
    # EI_t = E[max(0, f_best - Y)] where Y ~ t_df(mu, sigma)
    # This equals:
    #   (f_best - mu) * P(Y < f_best) + sigma * t_pdf(z; df) * (1 + z^2/df)^(-1/2) * ... 
    # Actually, the standard EI formula with t-distribution:
    #   EI = (f_best - mu) * F_t(z) + sigma * f_t(z) * (1 + z^2/df) / (df - 1)
    # where F_t and f_t are CDF and PDF of t-distribution with df degrees of freedom.
    #
    # More precisely, for Y ~ N(mu, sigma^2) transformed to t:
    # If Z ~ t_df, then Y = mu + sigma * Z / sqrt(df/(df-1)) approximately,
    # but the cleaner formulation is:
    #
    # Let Z ~ t(df). Then Y = mu + sigma * Z.
    # E[max(0, f_best - Y)] = E[max(0, f_best - mu - sigma*Z)]
    #                          = (f_best - mu) * P(Z < (f_best-mu)/sigma) 
    #                            + sigma * E[Z * 1{Z < z0}] where z0 = (f_best-mu)/sigma
    #
    # For t-distribution, E[Z * 1{Z < z0}] = -f_t(z0) * (1 + z0^2/df) / (df - 1)
    # (this is a known identity for t-distributions)
    
    z0 = z  # (f_best - mu) / sigma
    
    # CDF of t-distribution
    F_t = t_dist.cdf(z0, df)
    
    # PDF of t-distribution
    f_t = t_dist.pdf(z0, df)
    
    # E[Z * 1{Z < z0}] for t-distribution = -f_t(z0) * (1 + z0^2/df) / (df - 1)
    E_Z_neg = -f_t * (1.0 + z0**2 / df) / (df - 1.0)
    
    # EI = (f_best - mu) * F_t + sigma * E_Z_neg
    # But note: E[max(0, f_best - Y)] = (f_best - mu)*F_t(z0) - sigma * E[Z*1{Z>z0}]
    # Actually let's be more careful:
    # max(0, f_best - Y) = max(0, (f_best - mu) - sigma*Z)
    # = (f_best - mu - sigma*Z) * 1{Z < z0}
    # E = (f_best - mu)*P(Z < z0) - sigma*E[Z*1{Z < z0}]
    # = (f_best - mu)*F_t(z0) - sigma * E[Z*1{Z < z0}]
    # And E[Z*1{Z < z0}] = -f_t(z0)*(1+z0^2/df)/(df-1)
    # So EI = (f_best - mu)*F_t(z0) + sigma*f_t(z0)*(1+z0^2/df)/(df-1)
    
    EI = improvement * F_t + sigma_safe * f_t * (1.0 + z0**2 / df) / (df - 1.0)
    
    # Ensure scores are finite
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
# EVOLVE-BLOCK-END
