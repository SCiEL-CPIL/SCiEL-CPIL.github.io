# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _quantile_lcb_score(mu, sigma, f_best, cost, quantile=0.05, cost_scale=1.5,
                        cost_steepness=3.0, noise_floor=1e-9):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound.
    
    Uses the posterior predictive quantile (e.g., 5th percentile) as the
    'pessimistic' value, then applies a sigmoidal cost discount that
    aggressively penalizes high-cost candidates.
    
    The quantile approach is more robust than linear sigma scaling because
    it directly models the tail of the predictive distribution.
    """
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    # Quantile of the predictive distribution (lower = more pessimistic)
    # For a normal distribution, the q-quantile is mu + sigma * z_q
    # where z_q = norm.ppf(q)
    z_q = norm.ppf(quantile)  # negative for quantile < 0.5
    q_value = mu + z_q * sigma_safe
    
    # Sigmoidal cost discount: smoothly transitions from ~1 (low cost) to ~0 (high cost)
    # cost is in [1, 2] since cost = 1 + mean(x) and x in [0,1]^3
    # We want to penalize higher costs more aggressively
    cost_center = 1.5  # midpoint of cost range
    sigmoid_arg = cost_steepness * (cost_safe - cost_center)
    cost_discount = 1.0 / (1.0 + np.exp(sigmoid_arg))
    
    # Score: higher is better. We want low q_value (good minimum) and low cost.
    # Transform: negative q_value gives higher score for better minima
    # Multiply by cost discount to penalize expensive candidates
    score = (-q_value) * cost_discount
    
    return score


def _adaptive_quantile_lcb(mu, sigma, f_best, cost, noise_floor=1e-9):
    """
    Adaptive quantile LCB that adjusts the quantile based on the current
    state of the search. Early on, use a more exploratory quantile (e.g., 0.15);
    later, switch to a more exploitative quantile (e.g., 0.02).
    
    Also adapts the cost discount steepness based on the spread of costs.
    """
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    # Estimate progress: how close are we to the known best?
    # If mu is close to f_best everywhere, we're in exploitation mode
    rel_uncertainty = np.mean(sigma_safe / np.maximum(np.abs(mu), noise_floor))
    
    # Adapt quantile: more exploration when uncertainty is high
    # quantile ranges from 0.20 (high exploration) to 0.02 (high exploitation)
    base_quantile = 0.10
    # If relative uncertainty is high, explore more (higher quantile = less pessimistic)
    # If relative uncertainty is low, exploit more (lower quantile = more pessimistic)
    if rel_uncertainty > 1.0:
        quantile = min(0.20, base_quantile + 0.1 * (rel_uncertainty - 1.0))
    else:
        quantile = max(0.02, base_quantile - 0.08 * (1.0 - rel_uncertainty))
    
    # Adapt cost steepness: if costs are spread out, be more aggressive
    cost_spread = np.std(cost_safe)
    cost_steepness = 2.0 + 3.0 * cost_spread
    
    z_q = norm.ppf(quantile)
    q_value = mu + z_q * sigma_safe
    
    cost_center = 1.5
    sigmoid_arg = cost_steepness * (cost_safe - cost_center)
    cost_discount = 1.0 / (1.0 + np.exp(sigmoid_arg))
    
    score = (-q_value) * cost_discount
    
    return score


def _rank_based_score(mu, sigma, f_best, cost, noise_floor=1e-9):
    """
    Rank-based score that combines quantile LCB with a cost-aware ranking.
    
    This approach is robust to scale differences between the objective
    and cost by using ranks.
    """
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    # Quantile value
    quantile = 0.08
    z_q = norm.ppf(quantile)
    q_value = mu + z_q * sigma_safe
    
    # Rank-based cost penalty: higher cost rank = larger penalty
    # Convert cost to rank in [0, 1]
    cost_ranks = np.argsort(np.argsort(cost_safe)).astype(np.float64)
    n = len(cost_safe)
    if n > 1:
        cost_ranks = cost_ranks / (n - 1)
    else:
        cost_ranks = np.zeros_like(cost_safe)
    
    # Score: low q_value is good, low cost rank is good
    # Use exponential weighting to emphasize the best candidates
    q_value_rank = np.argsort(np.argsort(q_value)).astype(np.float64)
    if n > 1:
        q_value_rank = 1.0 - q_value_rank / (n - 1)  # invert: best q_value gets rank 1
    else:
        q_value_rank = np.ones_like(cost_safe)
    
    score = q_value_rank * (1.0 - 0.5 * cost_ranks)
    
    return score


def _calibrate_quantile(mu, sigma, f_best, cost, time_budget=5.0, noise_floor=1e-9):
    """
    Calibrate the quantile and cost parameters using a proxy objective.
    """
    t0 = time.time()
    
    quantile_vals = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    cost_center_vals = [1.3, 1.4, 1.5, 1.6]
    cost_steepness_vals = [1.5, 2.0, 3.0, 4.0, 5.0]
    
    top_k = min(20, max(3, len(mu) // 200))
    
    best_proxy = -np.inf
    best_params = (0.08, 1.5, 3.0)  # defaults
    
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    for q in quantile_vals:
        for cc in cost_center_vals:
            for cs in cost_steepness_vals:
                if time.time() - t0 > time_budget:
                    return best_params
                
                z_q = norm.ppf(q)
                q_value = mu + z_q * sigma_safe
                
                sigmoid_arg = cs * (cost_safe - cc)
                cost_discount = 1.0 / (1.0 + np.exp(sigmoid_arg))
                
                score = (-q_value) * cost_discount
                
                if not np.all(np.isfinite(score)):
                    continue
                
                top_idx = np.argsort(score)[-top_k:]
                mean_q = np.mean(q_value[top_idx])
                mean_cost = np.mean(cost_safe[top_idx])
                
                # Proxy: want low q_value (good minimum), low cost
                # Scale to make terms comparable
                q_scale = np.std(q_value) + 1e-9
                cost_scale = np.std(cost_safe) + 1e-9
                
                proxy = -(mean_q - np.mean(q_value)) / q_scale - 0.5 * (mean_cost - np.mean(cost_safe)) / cost_scale
                
                if proxy > best_proxy:
                    best_proxy = proxy
                    best_params = (q, cc, cs)
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Hartmann 3D minimization.
    Uses Quantile-Based Cost-Discounted Lower Confidence Bound.
    """
    t_start = time.time()
    deadline = t_start + 310.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters
    quantile, cost_center, cost_steepness = _calibrate_quantile(
        mu_safe, sigma_safe, f_best, cost_safe, time_budget=5.0
    )
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple quantile LCB with default params
        z_q = norm.ppf(0.08)
        q_value = mu_safe + z_q * sigma_safe
        sigmoid_arg = 3.0 * (cost_safe - 1.5)
        cost_discount = 1.0 / (1.0 + np.exp(sigmoid_arg))
        scores = (-q_value) * cost_discount
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute quantile LCB score with calibrated parameters
    z_q = norm.ppf(quantile)
    q_value = mu_safe + z_q * sigma_safe
    
    sigmoid_arg = cost_steepness * (cost_safe - cost_center)
    cost_discount = 1.0 / (1.0 + np.exp(sigmoid_arg))
    
    scores = (-q_value) * cost_discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
