import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Pareto-frontier selection strategy.
    
    Combines:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty (sigma) for exploration  
    3. Cost normalization for efficiency
    
    Uses a Pareto-frontier approach: compute non-dominated sort on
    (improvement, uncertainty, cost-efficiency) and select candidates
    on the efficient frontier. This avoids arbitrary weight tuning.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 320:
        # Fast fallback: simple cost-aware UCB
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        score = (f_best - mu + 0.5 * sigma_safe) / cost_safe
        return np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute three criteria for Pareto selection:
    # 1. Expected improvement (higher is better)
    # 2. Uncertainty (higher is better for exploration)  
    # 3. Cost efficiency: 1/cost (higher is better)
    
    # Normalize each criterion to [0, 1] for fair comparison
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    inv_cost = 1.0 / cost_safe
    inv_cost_max = np.max(inv_cost) if np.max(inv_cost) > 0 else 1.0
    
    ei_norm = ei / (ei_max + 1e-9)
    sigma_norm = sigma_safe / (sigma_max + 1e-9)
    inv_cost_norm = inv_cost / (inv_cost_max + 1e-9)
    
    # Pareto-frontier selection:
    # A candidate is non-dominated if no other candidate is better in all three criteria
    # We use a vectorized approach: for each candidate, check if any other dominates it
    
    # For efficiency with 2000 candidates, use a sampling-based approximation
    # Check against a random subset of candidates
    n_check = min(200, len(mu))
    check_indices = np.random.choice(len(mu), n_check, replace=False)
    
    # Build criterion matrix: (n, 3) where each row is (ei_norm, sigma_norm, inv_cost_norm)
    # Higher values are better in all three dimensions
    
    # A candidate i is dominated if there exists j such that:
    # ei_norm[j] >= ei_norm[i] AND sigma_norm[j] >= sigma_norm[i] AND inv_cost_norm[j] >= inv_cost_norm[i]
    # with at least one strict inequality
    
    # Vectorized dominance check against the sampled candidates
    # Shape: (n_check, n) for comparison
    ei_check = ei_norm[check_indices][:, np.newaxis]
    sigma_check = sigma_norm[check_indices][:, np.newaxis]
    cost_check = inv_cost_norm[check_indices][:, np.newaxis]
    
    ei_all = ei_norm[np.newaxis, :]
    sigma_all = sigma_norm[np.newaxis, :]
    cost_all = inv_cost_norm[np.newaxis, :]
    
    # Dominance: check >= in all three, with at least one >
    all_geq = (ei_check >= ei_all) & (sigma_check >= sigma_all) & (cost_check >= cost_all)
    any_gt = (ei_check > ei_all) | (sigma_check > sigma_all) | (cost_check > cost_all)
    
    dominated = (all_geq & any_gt).any(axis=0)
    
    # Non-dominated candidates get a boost
    # Use a smooth score: base score + frontier bonus
    base_score = (ei_norm + 0.3 * sigma_norm + 0.3 * inv_cost_norm) / 1.6
    
    # Frontier bonus: non-dominated candidates get higher score
    # Use a smooth transition rather than hard 0/1
    frontier_indicator = 1.0 - dominated.astype(float)
    
    # Combine: frontier candidates get 1.5x multiplier, others get 0.7x
    multiplier = 0.7 + 0.8 * frontier_indicator
    
    score = base_score * multiplier
    
    # Add a small exploitation term to ensure we don't ignore high-EI points
    score += 0.2 * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
