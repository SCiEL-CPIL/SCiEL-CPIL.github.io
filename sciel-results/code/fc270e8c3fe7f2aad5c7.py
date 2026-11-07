import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (griewank 2d).
    
    Uses Expected Improvement with cost normalization, plus a small UCB term
    to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP posterior
    sigma: array of shape (2000,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement for minimization
    # Improvement is (f_best - y), so we want E[max(f_best - y, 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization: divide by cost
    # cost is in [1, 2] since cost = 1 + mean(x_scaled) and x_scaled in [0,1]
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Add a small UCB-like exploration term to encourage exploration
    # This helps escape local optima in the griewank function
    # Use a fixed beta that balances exploration/exploitation
    beta = 0.1
    ucb_term = beta * sigma_safe / cost_safe
    
    # Combined score: cost-aware EI + small exploration term
    scores = ei_cost + ucb_term
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
