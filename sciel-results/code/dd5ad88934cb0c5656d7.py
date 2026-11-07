"""Rewritten program: cost-aware acquisition with Student-t robustness."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost,
    with a Student-t inspired variance adjustment for robustness.
    
    We model the uncertainty as a Student-t distribution by scaling the
    predicted standard deviation to match the variance of a t-distribution
    with specific degrees of freedom. This increases exploration in uncertain
    regions compared to standard Normal EI.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Student-t robustness parameter
    # Degrees of freedom for the t-distribution approximation.
    # Lower df -> heavier tails -> larger effective variance.
    df = 5.0
    
    # Scale factor to match the variance of a Student-t distribution
    # Var(t) = sigma^2 * df / (df - 2)
    # So, sigma_eff = sigma * sqrt(df / (df - 2))
    scale_factor = np.sqrt(df / (df - 2.0))
    
    # Effective standard deviation incorporating Student-t uncertainty
    sigma_eff = sigma_safe * scale_factor
    
    # Standard normal z-score for EI using the effective sigma
    z = (f_best - mu) / sigma_eff
    
    # Expected Improvement (EI) calculated with the effective sigma
    # This accounts for the heavier tails of the Student-t model
    ei = (f_best - mu) * norm.cdf(z) + sigma_eff * norm.pdf(z)
    
    # Cost-aware scoring: EI per unit cost
    # This prioritizes candidates that offer high expected improvement relative to their evaluation cost
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
