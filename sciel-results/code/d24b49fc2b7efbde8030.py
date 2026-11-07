import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) with
    adaptive exploration via a variance-weighted UCB term.
    
    Args:
        mu: Array of GP means, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: The best observed function value (scalar)
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        scores: Array of acquisition scores, shape (N,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization: lower is better
    # Use a moderate kappa for exploration
    kappa = 1.0
    ucb = mu - kappa * sigma_safe
    
    # Cost-normalized scores
    # EI: higher is better (more expected improvement)
    # UCB: lower is better, so negate to make higher = better
    score_ei = ei / cost_safe
    score_ucb = -ucb / cost_safe
    
    # Combine with weights favoring exploitation early, exploration later
    # Use a simple weighted sum
    w_ei = 0.8
    w_ucb = 0.2
    
    scores = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
