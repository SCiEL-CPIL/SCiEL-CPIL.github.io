# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.
    
    Uses a Quantile-Based Cost-Discounted Lower Confidence Bound (LCB)
    that maps the posterior predictive to a dynamic quantile and applies
    a sigmoidal cost discount to aggressively prune high-cost candidates.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    
    # Quantile-based LCB:
    # Instead of using LCB = mu - beta * sigma, we use a dynamic quantile
    # approach where the quantile level adapts based on the relative
    # uncertainty and the current best value.
    
    # Compute z-scores relative to f_best
    z = (f_best - mu_safe) / sigma_safe
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Dynamic quantile: use a lower quantile for points with high uncertainty
    # and closer to f_best, higher quantile for points far from f_best
    # The quantile level q in [0, 1] determines which percentile to use
    # For minimization, we want a low quantile (optimistic) for exploration
    
    # Base quantile: start at 5th percentile (0.05) for aggressive exploration
    # Adjust based on how many points are close to f_best
    close_to_best = np.sum(z > 0) / n  # fraction of points that could improve
    if close_to_best < 0.01:
        base_quantile = 0.02  # very aggressive when few points can improve
    elif close_to_best < 0.05:
        base_quantile = 0.05
    elif close_to_best < 0.2:
        base_quantile = 0.1
    else:
        base_quantile = 0.15
    
    # For each point, compute the quantile value of N(mu, sigma^2)
    # Using the inverse CDF: quantile_value = mu + sigma * Phi^{-1}(q)
    # For minimization, we want a low quantile, so q is small
    # But we also want to account for the fact that points with high
    # uncertainty should be explored more
    
    # Adaptive quantile per point: points with higher sigma get slightly
    # lower quantiles (more aggressive exploration)
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-9:
        sigma_max = 1.0
    sigma_relative = sigma_safe / sigma_max  # [0, 1]
    
    # Quantile adjustment: higher uncertainty -> lower quantile
    # q_i = base_quantile * (1 - 0.5 * sigma_relative)
    quantile_levels = base_quantile * (1 - 0.5 * sigma_relative)
    quantile_levels = np.clip(quantile_levels, 0.001, 0.5)
    
    # Compute quantile values using the normal distribution
    # For each point, compute mu_i + sigma_i * Phi^{-1}(q_i)
    # We need to compute Phi^{-1} for different quantiles, so we use
    # the fact that for a normal distribution, the quantile function
    # is mu + sigma * z where z is the standard normal quantile
    
    # Since quantile_levels vary per point, we compute this vectorized
    # using the inverse CDF (ppf) of the standard normal
    z_quantiles = norm.ppf(quantile_levels)  # negative values for low quantiles
    
    # Quantile-based LCB value
    quantile_lcb = mu_safe + sigma_safe * z_quantiles
    
    # The score is negative of the quantile LCB (higher is better for minimization)
    quantile_scores = -quantile_lcb
    
    # Cost-aware sigmoidal discount:
    # Cost is in [1, 2]. We apply a sigmoidal discount that is close to 1
    # for low costs and drops sharply for high costs.
    # discount = 1 / (1 + exp(k * (cost - c0)))
    # where k controls the steepness and c0 is the midpoint
    
    # Normalize cost to [0, 1] for the sigmoid
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_normalized = np.zeros_like(cost_safe)
    else:
        cost_normalized = (cost_safe - cost_min) / cost_range
    
    # Sigmoidal discount: steep drop for high-cost candidates
    # k = 8 for aggressive pruning, c0 = 0.5 for midpoint
    k_sigmoid = 8.0
    c0_sigmoid = 0.5
    sigmoid_discount = 1.0 / (1.0 + np.exp(k_sigmoid * (cost_normalized - c0_sigmoid)))
    
    # Apply the discount to the scores
    # The discount is in [0, 1], so it reduces scores for high-cost points
    cost_discounted_scores = quantile_scores * sigmoid_discount
    
    # Ensure finite
    cost_discounted_scores = np.where(np.isfinite(cost_discounted_scores), cost_discounted_scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return cost_discounted_scores
    
    # Final safety
    cost_discounted_scores = np.where(np.isfinite(cost_discounted_scores), cost_discounted_scores, 0.0)
    
    return cost_discounted_scores
# EVOLVE-BLOCK-END
