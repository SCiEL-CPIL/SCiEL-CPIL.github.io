"""Rewritten program: Quantile-Based Cost-Discounted Lower Confidence Bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound.
    
    Instead of using standard EI (which linearly scales with sigma), we map the
    posterior predictive distribution to a dynamic quantile (e.g., 5th percentile)
    to capture the lower tail of the distribution. We then apply a non-linear
    sigmoidal cost discount to aggressively penalize high-cost candidates.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior predictive distribution.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior predictive distribution.
    f_best : float
        Best observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic quantile: use the 5th percentile of the predictive distribution
    # This is equivalent to mu - z * sigma where z is the 95th percentile of the standard normal
    # (since we want the 5th percentile, we use z = norm.ppf(0.95) for the lower tail)
    # For the 5th percentile: x_0.05 = mu + sigma * norm.ppf(0.05)
    # norm.ppf(0.05) ≈ -1.6449
    quantile_z = norm.ppf(0.05)  # ≈ -1.6449
    
    # Compute the quantile-based value (lower is better for the objective)
    # The quantile value represents a "lower confidence" estimate
    quantile_value = mu + quantile_z * sigma_safe
    
    # We want to minimize, so lower quantile_value is better
    # Convert to a "score" where higher is better: negative of quantile_value
    base_score = -quantile_value
    
    # Apply sigmoidal cost discount
    # Cost ranges from 1.0 (center) to 2.0 (corners)
    # We want to heavily discount high-cost candidates
    # Sigmoid: discount = 1 / (1 + exp(k * (cost - cost_center)))
    # We want the discount to be close to 1 for low cost and close to 0 for high cost
    # Let's center at cost = 1.5 (midpoint of [1, 2])
    cost_center = 1.5
    cost_scale = 0.3  # Controls the steepness of the sigmoid
    
    # Compute sigmoidal discount
    # For cost < cost_center, discount > 0.5; for cost > cost_center, discount < 0.5
    # We want to amplify the difference, so use a steeper sigmoid
    k = 8.0  # Steepness parameter
    discount = 1.0 / (1.0 + np.exp(k * (cost - cost_center)))
    
    # Combine base score with cost discount
    # Multiply the base score by the discount to penalize high-cost candidates
    scores = base_score * discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
