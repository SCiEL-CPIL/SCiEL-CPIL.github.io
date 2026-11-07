"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using smooth max of EI and UCB per unit cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_term = np.maximum(ei_term, 0.0)
    
    score_ei = ei_term / cost_safe
    
    # UCB score: rewards low mu and high sigma
    beta = 2.0
    score_ucb = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Use smooth max (log-sum-exp) to combine scores smoothly
    # This avoids the kink in np.maximum and may help optimization
    # Scale factor for smoothness
    temp = 0.1
    
    # For numerical stability in log-sum-exp
    max_score = np.maximum(score_ei, score_ucb)
    # Avoid log of zero
    safe_max = np.maximum(max_score, 1e-10)
    
    log_sum = safe_max + temp * np.log(np.exp((score_ei - safe_max) / temp) + np.exp((score_ucb - safe_max) / temp))
    
    scores = log_sum
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
