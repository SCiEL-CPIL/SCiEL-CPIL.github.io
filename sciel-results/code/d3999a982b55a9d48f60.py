"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a quantile-based heuristic.
    
    Instead of using the Probability of Improvement (PI) which saturates,
    we use the improvement over a lower quantile (10th percentile) of the
    predictive distribution, normalized by the evaluation cost.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the 10th percentile of the predictive distribution.
    # For a normal distribution N(mu, sigma^2), the 10th percentile is:
    # mu + sigma * Phi^{-1}(0.1)
    # Phi^{-1}(0.1) is approximately -1.28155
    z_quantile = -1.28155
    v_low = mu + z_quantile * sigma_safe
    
    # Calculate the potential improvement over the current best.
    # We only care about points where the lower quantile is below f_best.
    # This ensures we are looking for points with a high probability of being significantly better.
    improvement = np.maximum(0.0, f_best - v_low)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score: Ratio of quantile-based improvement to cost
    scores = improvement / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
