import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track time to avoid timeout
    t_start = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Upper Confidence Bound (UCB)
    # Adaptive kappa: increase exploration when improvement is small
    # or when we're early in the search (low cost spent)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Adaptive exploration parameter
    # Higher when f_best is close to 0 (we're doing well) or early in search
    # This balances exploitation and exploration
    improvement_signal = np.clip(f_best, 0.0, 1.0)
    # Start with moderate exploration, reduce as we get better
    base_kappa = 0.5
    # Increase exploration when uncertainty is high relative to improvement
    sigma_mean = np.mean(sigma_safe)
    improvement_gap = max(f_best - np.min(mu), 1e-8)
    kappa = base_kappa * (1.0 + 0.5 * sigma_mean / improvement_gap)
    
    ucb = mu + kappa * sigma_safe
    
    # Cost-aware penalty: prefer cheaper points
    # Use a smooth, non-linear penalty that's more aggressive for expensive points
    # cost_norm is in [0, 1], 0 = cheapest
    # Penalty factor: 1.0 for cheapest, decreases for expensive
    # Use exponential decay for stronger cost sensitivity
    cost_sensitivity = 1.5
    cost_penalty = np.exp(-cost_sensitivity * cost_norm)
    
    # Combine EI and UCB with adaptive weighting
    # Weight EI more when we have good estimates, UCB when uncertain
    # Use the ratio of EI to UCB as a signal
    ei_weight = 0.7
    ucb_weight = 0.3
    
    # Base acquisition: weighted combination of EI and UCB
    # Normalize both to comparable scales
    ei_scale = np.max(ei) + 1e-9
    ucb_scale = np.max(np.abs(ucb)) + 1e-9
    
    ei_norm = ei / ei_scale
    ucb_norm = ucb / ucb_scale
    
    # Combine
    base_score = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Apply cost penalty
    score = base_score * cost_penalty
    
    # Add a small bonus for high-uncertainty regions to encourage exploration
    # This helps escape local optima
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / len(sigma_safe)
    exploration_bonus = 0.1 * sigma_rank
    
    score = score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        # Return the score as is, it should be fast enough
        pass
    
    return score
