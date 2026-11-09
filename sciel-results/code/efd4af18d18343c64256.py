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
    
    # Dynamic Quantile Strategy:
    # Blend between a conservative quantile (25th percentile, z=-0.67) and
    # an aggressive quantile (1st percentile, z=-2.33) based on uncertainty.
    # High uncertainty regions get more aggressive exploration (lower z).
    z_conservative = norm.ppf(0.25)  # ≈ -0.6745
    z_aggressive = norm.ppf(0.01)    # ≈ -2.3263
    
    # Weight based on sigma: high sigma -> weight close to 1 (aggressive)
    # Use a saturation function to keep it bounded.
    sigma_weight = sigma_safe / (sigma_safe + 0.5)
    
    # Interpolate z-score
    quantile_z = z_conservative + (z_aggressive - z_conservative) * sigma_weight
    
    # Compute the quantile-based value (lower is better for the objective)
    quantile_value = mu + quantile_z * sigma_safe
    
    # We want to minimize, so lower quantile_value is better
    # Convert to a "score" where higher is better: negative of quantile_value
    base_score = -quantile_value
    
    # Non-linear Sigmoidal Cost Penalty:
    # Cost ranges from 1.0 to 2.0.
    # We apply a penalty that is negligible for low costs and steep for high costs.
    # Center the sigmoid around 1.4 with a steep slope k=10.
    cost_center = 1.4
    cost_steepness = 10.0
    max_penalty = 2.0
    
    # Sigmoid input
    x_cost = cost_steepness * (cost - cost_center)
    # Sigmoid output (0 to 1)
    sigmoid_cost = 1.0 / (1.0 + np.exp(-x_cost))
    
    # Penalty is proportional to sigmoid output
    cost_penalty = max_penalty * sigmoid_cost
    
    # Combine base score with cost penalty
    scores = base_score - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
