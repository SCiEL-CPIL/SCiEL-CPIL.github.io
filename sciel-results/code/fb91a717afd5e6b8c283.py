"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Lower Confidence Bound (LCB) for exploration-exploitation balance.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
        
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Standardized improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Expected Improvement (EI)
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) - favors exploration
    # LCB = mu - beta * sigma (we want low values, so lower is better)
    # For maximization of acquisition: -LCB = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both components to comparable scales
    # Use cost-normalized versions
    ei_cost = ei / cost_safe
    lcb_cost = lcb_score / cost_safe
    
    # Combine: weighted sum with emphasis on EI for exploitation,
    # LCB for exploration when uncertainty is high
    # Use a dynamic weight: more exploration when sigma is high relative to improvement
    sigma_ratio = sigma_safe / (np.abs(improvement) + 1e-8)
    weight_lcb = np.clip(0.3 * sigma_ratio, 0.0, 0.5)
    weight_ei = 1.0 - weight_lcb
    
    # Final score: higher is better
    score = weight_ei * ei_cost + weight_lcb * lcb_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
