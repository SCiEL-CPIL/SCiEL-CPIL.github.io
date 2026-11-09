"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _safe_divide(a, b, eps=1e-9):
    """Safe element-wise division with epsilon guard."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a UCB-style exploration term,
    normalized by the square root of the cost. This balances exploitation
    (EI) with exploration (uncertainty) while mitigating the strong bias
    towards low-cost corners inherent in simple cost-normalization.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = _safe_divide(diff, sigma)
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for very large values
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # UCB term: encourages exploration of high-uncertainty regions.
    # beta is tuned to be significant enough to escape local minima in Griewank
    # but not so large as to ignore the mean prediction entirely.
    beta = 1.0
    ucb = beta * sigma
    
    # Combined objective: EI + UCB
    # This creates a hybrid acquisition that is greedy when uncertainty is low
    # but exploratory when uncertainty is high.
    combined = ei + ucb
    
    # Cost-aware: divide by square root of cost.
    # sqrt(cost) is in [1, 1.414] for cost in [1, 2].
    # Dividing by sqrt(cost) reduces the penalty for evaluating in higher-cost
    # (central) regions compared to dividing by cost directly, which helps
    # the search reach the global minimum located in the center of the domain.
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = np.sqrt(cost_safe)
    
    score = _safe_divide(combined, cost_factor, eps=1e-9)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
