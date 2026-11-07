"""Acquisition function for Bayesian optimization of hartmann 3d using Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """Expected Improvement acquisition function.
    
    Maximizes the expected reduction in the minimum observed value.
    EI(x) = E[max(0, f_best - f(x))]
    where f(x) ~ N(mu(x), sigma(x)^2)
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidate points
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        EI values, higher is better
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate the improvement I = (f_best - mu) / sigma
    # Note: Since we are minimizing, if mu < f_best, the mean improvement is already positive.
    # The formula handles the case where mu > f_best (exploration) and mu < f_best (exploitation).
    
    I = (f_best - mu) / sigma_safe
    
    # EI = sigma * (I * Phi(I) + phi(I))
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    
    cdf_I = norm.cdf(I)
    pdf_I = norm.pdf(I)
    
    scores = sigma_safe * (I * cdf_I + pdf_I)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
