"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Upper Confidence Bound, with adaptive exploration.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.shape != sigma.shape or mu.shape != cost.shape:
        raise ValueError("mu, sigma, and cost must have the same shape")
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Upper Confidence Bound (UCB) for minimization
    # UCB = -mu + beta * sigma
    # Use adaptive beta based on the scale of the problem
    beta = 2.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weight
    # Use a weighted combination that balances exploitation (EI) and exploration (UCB)
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-9
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1e-9
    
    # Weight: 0.6 for EI (exploitation), 0.4 for UCB (exploration)
    w_ei = 0.6
    w_ucb = 0.4
    
    # Combined score before cost adjustment
    combined = w_ei * (ei / ei_max) + w_ucb * (ucb / ucb_max)
    
    # Adjust for cost: divide by cost to get score per unit cost
    # But we want to preserve the relative ordering, so use a softer cost adjustment
    # score = combined / cost_safe
    # However, this can be too aggressive. Use a power adjustment.
    score = combined / (cost_safe ** 0.5)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
