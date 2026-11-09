"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (cEI).
    
    This approach maximizes the expected reduction in the objective function value
    (regret) per unit of cost. This is a structural change from LCB, focusing on
    the probabilistic gain of each evaluation normalized by its expense.
    
    Parameters:
    mu: (2000,) array of GP means.
    sigma: (2000,) array of GP standard deviations.
    f_best: float, best observed value.
    cost: (2000,) array of evaluation costs.
    
    Returns:
    (2000,) array of scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # In GP, sigma should be > 0, but numerical errors can cause 0.
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate Z-score for Expected Improvement
    # We want P(f(x) < f_best) and E[f_best - f(x) | f(x) < f_best]
    # Let Y = f(x) ~ N(mu, sigma^2)
    # Improvement I = max(0, f_best - Y)
    # E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / safe_sigma
    
    # Standard Normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + safe_sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost Adjustment
    # We divide by cost to maximize improvement per unit cost.
    # Cost is in [1, 2].
    safe_cost = np.maximum(cost, eps)
    
    scores = ei / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
