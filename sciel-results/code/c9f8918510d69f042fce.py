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
    
    # Dynamic quantile: use the 1st percentile of the predictive distribution
    # This is more aggressive exploration than 5th percentile.
    # x_0.01 = mu + sigma * norm.ppf(0.01)
    # norm.ppf(0.01) ≈ -2.3263
    quantile_z = norm.ppf(0.01)  # ≈ -2.3263
    
    # Compute the quantile-based value (lower is better for the objective)
    # The quantile value represents a "lower confidence" estimate
    quantile_value = mu + quantile_z * sigma_safe
    
    # We want to minimize, so lower quantile_value is better
    # Convert to a "score" where higher is better: negative of quantile_value
    base_score = -quantile_value
    
    # Apply linear cost penalty
    # Cost ranges from 1.0 to 2.0.
    # We penalize higher costs linearly to avoid the ordering inversion
    # issue present in multiplicative discounts for negative scores.
    # Penalty coefficient lambda = 0.5 balances exploration vs cost.
    cost_penalty = 0.5 * cost
    
    # Combine base score with cost penalty
    scores = base_score - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
