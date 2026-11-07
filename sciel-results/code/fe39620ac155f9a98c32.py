"""Rewritten program with global budget-aware scaling strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using global budget-aware scaling.
    
    Instead of point-wise division by cost, we compute the expected improvement
    and then apply a global penalty based on the ratio of each candidate's cost
    to the mean cost of the batch. This shifts the exploration-exploitation
    balance based on remaining budget rather than local cost ratios.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean of the GP posterior in objective units.
    sigma : ndarray of shape (2000,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed value so far (lowest).
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Compute expected improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Handle cases where improvement is negative (mu > f_best)
    # In those cases, EI should be 0 (or very small)
    ei = np.zeros_like(mu)
    positive_mask = improvement > 0
    if np.any(positive_mask):
        ei[positive_mask] = (
            improvement[positive_mask] * norm.cdf(z[positive_mask]) +
            sigma_safe[positive_mask] * norm.pdf(z[positive_mask])
        )
    
    # Global budget-aware scaling with dynamic penalty scaling
    # We want the cost penalty to be comparable in magnitude to the EI signal.
    
    # Calculate statistics of EI and Cost to determine the penalty weight
    ei_std = np.std(ei)
    cost_std = np.std(cost)
    mean_cost = np.mean(cost)
    
    # If variance is too low, use a default scale to avoid division by zero or extreme values
    if ei_std < eps:
        ei_std = 1.0
    if cost_std < eps:
        cost_std = 1.0
    
    # Dynamic lambda: Scale the cost penalty so its variance matches EI's variance
    # This ensures that cost matters proportionally to the uncertainty/improvement signal
    lambda_penalty = ei_std / cost_std
    
    # Penalty term: linear deviation from mean cost
    # Candidates more expensive than average are penalized, cheaper are rewarded
    penalty = lambda_penalty * (cost - mean_cost)
    
    # Final score: EI minus scaled cost penalty
    scores = ei - penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
