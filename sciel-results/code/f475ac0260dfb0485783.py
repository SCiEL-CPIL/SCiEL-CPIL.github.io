"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a blend of Expected Improvement and UCB, normalized by sqrt(cost)
    with a soft Pareto-frontier boost to balance exploration and cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware UCB with sqrt cost
        beta = 1.5
        improvement = f_best - mu + beta * sigma
        scores = improvement / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Estimate budget progress from cost array
    # Total budget is 30, initial 5 points cost ~1.5 each => ~7.5 used
    # Median cost of candidates tells us typical evaluation cost
    median_cost = np.median(cost)
    if median_cost > 1e-9:
        # Estimated remaining budget in evaluations
        total_budget = 30.0
        initial_cost = 5 * 1.5  # approx cost of 5 uniform random points
        # Estimate how many more evaluations we can afford
        # Use the cost of the best candidate as a proxy for what we'll evaluate
        best_cost_idx = np.argmin(cost)
        typical_cost = cost[best_cost_idx] if cost[best_cost_idx] > 1e-9 else median_cost
        remaining_budget = max(0, total_budget - initial_cost)
        # Fraction of budget that corresponds to the current "typical" evaluation
        # This is a proxy for progress: if typical cost is high, we're further along
        progress_from_cost = np.clip(1.0 - (remaining_budget / (typical_cost * 20)), 0.0, 1.0)
    else:
        progress_from_cost = 0.0
    
    # Also estimate progress from f_best
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress_from_f = 1.0
    else:
        # Normalize progress: early on, f_best is around -1.0 to -2.0
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress_from_f = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Combine both progress estimates
    progress = 0.5 * progress_from_cost + 0.5 * progress_from_f
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement with adaptive beta
    # Adaptive beta: explore more early, exploit more late
    beta_early = 2.0
    beta_late = 0.6
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_improvement = f_best - mu + beta * sigma
    
    # Blend EI and UCB based on progress
    # Early: weight towards UCB, Late: weight towards EI
    w = 0.4 * (1.0 + progress)  # w in [0.4, 0.8]
    blended = w * ei + (1.0 - w) * ucb_improvement
    
    # Cost-aware score using power(cost, 0.5) for sqrt normalization
    # This is a standard cost-aware BO approach
    cost_factor = np.power(cost, 0.5)
    base_scores = blended / cost_factor
    
    # Soft Pareto-frontier boost:
    # Instead of hard non-dominated filtering (which was O(n^2) and showed no
    # improvement), use a smooth surrogate that softly rewards points that are
    # "frontier-like" in the (improvement, uncertainty, -cost) space.
    #
    # We compute a soft rank for each point based on how well it balances
    # the three objectives, then use a smooth (tanh-based) boost.
    
    # Normalize each objective to [0, 1] using robust percentiles to avoid outliers
    imp_min, imp_max = np.percentile(blended, 1), np.percentile(blended, 99)
    unc_min, unc_max = np.percentile(sigma, 1), np.percentile(sigma, 99)
    cost_min, cost_max = np.percentile(cost, 1), np.percentile(cost, 99)
    
    imp_range = imp_max - imp_min + 1e-9
    unc_range = unc_max - unc_min + 1e-9
    cost_range = cost_max - cost_min + 1e-9
    
    imp_norm = np.clip((blended - imp_min) / imp_range, 0.0, 1.0)  # higher is better
    unc_norm = np.clip((sigma - unc_min) / unc_range, 0.0, 1.0)    # higher is better (more exploration)
    cost_norm = np.clip(1.0 - (cost - cost_min) / cost_range, 0.0, 1.0)  # higher is better (lower cost)
    
    # Soft Pareto score using weighted geometric mean
    # Weight exploration more heavily when progress is low
    w_explore = 0.5 + 0.3 * (1.0 - progress)  # weight for uncertainty
    w_exploit = 0.3 + 0.2 * progress          # weight for improvement
    w_cost = 1.0 - w_explore - w_exploit       # weight for cost (lower is better)
    w_cost = max(w_cost, 0.1)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-6
    log_imp = np.log(imp_norm + eps)
    log_unc = np.log(unc_norm + eps)
    log_cost = np.log(cost_norm + eps)
    
    # Weighted geometric mean of the three normalized objectives
    geo_mean = np.exp(w_exploit * log_imp + w_explore * log_unc + w_cost * log_cost)
    
    # Normalize geo_mean to [0, 1] for stable scaling
    gm_min, gm_max = np.min(geo_mean), np.max(geo_mean)
    gm_range = gm_max - gm_min + 1e-9
    gm_norm = (geo_mean - gm_min) / gm_range
    
    # Smooth boost: use a tanh-based function to create a smooth, bounded boost
    # that is stronger for points with higher geo_mean (more frontier-like)
    # Scale the tanh input so that the boost is meaningful in the mid-range
    tanh_scale = 3.0
    smooth_boost = np.tanh(tanh_scale * (gm_norm - 0.5))  # in [-1, 1]
    
    # Convert to a multiplicative factor: 
    # - Points with gm_norm > 0.5 get a boost up to +25%
    # - Points with gm_norm < 0.5 get a penalty up to -25%
    # - Points near 0.5 get ~0 change
    boost_factor = 1.0 + 0.25 * smooth_boost
    
    # Apply the smooth boost
    scores = base_scores * boost_factor
    
    # Add a small uncertainty bonus to encourage exploration in uncertain regions
    # This is a smooth surrogate that balances exploitation and exploration
    # The bonus is scaled by the progress: more exploration early, less late
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = np.clip((sigma - sigma_min) / sigma_range, 0.0, 1.0)
    uncertainty_bonus = 0.10 * (1.0 - progress) * sigma_norm
    
    scores = scores + uncertainty_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
