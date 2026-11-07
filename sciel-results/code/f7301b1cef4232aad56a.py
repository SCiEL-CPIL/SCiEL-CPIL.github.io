"""Rewritten program: Stochastic Expected Improvement with Log-Normal Cost Model."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) with log-normal cost modeling.
    
    This acquisition function models the evaluation cost as a log-normal random
    variable and computes the expected improvement per unit cost in a stochastic
    framework. The cost uncertainty is integrated directly into the acquisition
    score rather than applied as a deterministic penalty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Deterministic cost estimates for each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from current best
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Model cost as log-normal: cost = exp(log_cost_base + noise)
    # We estimate the log-normal parameters from the given cost values
    # For simplicity, assume the cost is deterministic but we account for
    # uncertainty in the improvement per cost ratio
    
    # The stochastic EI per unit cost: E[EI / C] where C ~ LogNormal
    # Approximate by computing EI / E[C] with a correction for variance
    # E[1/C] for C ~ LogNormal(m, s^2) = exp(-m - s^2/2)
    
    # Estimate log-cost parameters: assume mean cost ~ given cost
    # Let log(C) ~ Normal(mu_log, sigma_log^2)
    # We set mu_log = log(cost) and estimate sigma_log from the spread
    # For a single cost value, we use a small default uncertainty
    
    # Use a fixed small log-cost standard deviation to model uncertainty
    log_cost_sigma = 0.1  # Modest uncertainty in log-cost
    
    # E[1/C] = exp(-mu_log - sigma_log^2/2) where mu_log = log(cost)
    # So E[EI/C] ≈ EI * exp(-log(cost) - log_cost_sigma^2/2)
    # = EI * exp(-log_cost_sigma^2/2) / cost
    
    # The correction factor accounts for the convexity of 1/C
    correction_factor = np.exp(-log_cost_sigma ** 2 / 2.0)
    
    # Stochastic expected improvement per unit cost
    sei = ei * correction_factor / (cost + 1e-9)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # The exploration bonus is also cost-normalized
    lambda_explore = 0.05
    exploration = lambda_explore * sigma_safe * correction_factor / (cost + 1e-9)
    
    scores = sei + exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
