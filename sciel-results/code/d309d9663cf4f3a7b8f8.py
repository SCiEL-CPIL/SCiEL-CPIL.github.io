"""Rewritten cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using expected improvement (EI)
    with a robustness factor to handle the cost structure.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # We want to maximize EI per unit cost, but also consider that
    # higher cost points might be worth more if they have much higher EI
    # Use a soft normalization: EI / (cost * (1 + log(1 + EI)))
    # This down-weights expensive points with modest EI while still
    # allowing very high EI to overcome cost
    
    # Alternative: use EI / cost with a floor to prevent division issues
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary metric: EI per unit cost
    score = ei / cost_safe
    
    # Add a small exploration bonus for high uncertainty to ensure
    # we don't completely ignore uncertain regions
    # This helps escape local optima
    uncertainty_bonus = sigma_safe * 0.1
    
    # Combine: weighted combination of cost-aware EI and uncertainty
    # The weight balances exploitation (EI/cost) with exploration (uncertainty)
    score = score + uncertainty_bonus / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
