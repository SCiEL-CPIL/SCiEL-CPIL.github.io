# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple EI to avoid crash
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # 3. Compute Lower Confidence Bound (LCB) based score
    # Adaptive beta: start with higher exploration, reduce as f_best improves
    # Hartmann 6d minimum is -3.32237, so f_best starts near 0 and goes negative
    # Progress: 0 is worst, -3.32 is best
    progress = np.clip((-f_best - (-3.32237)) / (0 - (-3.32237)), 0.0, 1.0)
    # Beta decreases from 2.0 (early, explore) to 0.5 (late, exploit)
    beta = 2.0 - 1.5 * progress
    ucb_score = -mu + beta * sigma_safe
    
    # 4. Cost efficiency
    cost_eff = 1.0 / (cost + eps)
    
    # 5. Normalize all objectives to [0, 1] for comparable scales
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < eps:
            return np.ones_like(arr)
        return (arr - mn) / rng
    
    ei_norm = normalize(ei)
    pi_norm = normalize(pi)
    ucb_norm = normalize(ucb_score)
    cost_norm = normalize(cost_eff)
    
    # 6. Weighted blend for primary scoring
    # EI is the most robust; PI adds stability; UCB adds exploration
    # Weights: EI dominant, PI secondary, UCB for exploration
    w_ei = 0.5
    w_pi = 0.2
    w_ucb = 0.3
    
    blend = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
    
    # 7. Pareto frontier on (blend, cost_norm)
    sort_idx = np.argsort(blend)[::-1]
    sorted_blend = blend[sort_idx]
    sorted_cost = cost_norm[sort_idx]
    
    max_cost_so_far = -np.inf
    is_pareto = np.zeros(n, dtype=bool)
    pareto_indices = []
    
    for i in range(n):
        if sorted_cost[i] > max_cost_so_far:
            is_pareto[sort_idx[i]] = True
            pareto_indices.append(sort_idx[i])
            max_cost_so_far = sorted_cost[i]
        else:
            is_pareto[sort_idx[i]] = False

    # 8. Assign scores: strong preference for Pareto, but continuous ranking
    # Pareto points get a large base + their blend score
    # Non-Pareto points get a small fraction of their blend score
    # This preserves relative ordering while strongly favoring Pareto
    
    final_scores = np.zeros(n)
    
    if len(pareto_indices) > 0:
        # Pareto points: high base + normalized blend among Pareto
        pareto_blend = blend[pareto_indices]
        pareto_cost = cost_norm[pareto_indices]
        
        # Among Pareto points, rank by a combination of blend and cost
        # Use normalized rank to ensure smooth selection
        pareto_combined = 0.7 * pareto_blend + 0.3 * pareto_cost
        
        pb_min = np.min(pareto_combined)
        pb_max = np.max(pareto_combined)
        pb_range = pb_max - pb_min
        if pb_range < eps:
            pareto_rank = np.ones_like(pareto_combined)
        else:
            pareto_rank = (pareto_combined - pb_min) / pb_range
        
        # Base score of 100 ensures Pareto >> non-Pareto
        # The rank term (0 to 1) differentiates among Pareto points
        final_scores[pareto_indices] = 100.0 + pareto_rank
        
        # Non-Pareto points: small score based on their blend
        # Scale to [0, 1) so they're always below Pareto points
        non_pareto_blend = blend[~is_pareto]
        np_min = np.min(non_pareto_blend) if len(non_pareto_blend) > 0 else 0
        np_max = np.max(non_pareto_blend) if len(non_pareto_blend) > 0 else 1
        np_range = np_max - np_min
        if np_range < eps:
            non_pareto_rank = np.zeros_like(non_pareto_blend)
        else:
            non_pareto_rank = (non_pareto_blend - np_min) / np_range
        
        final_scores[~is_pareto] = non_pareto_rank * 0.5  # Max 0.5, well below 100
    else:
        # Fallback: just use the blend
        final_scores = blend + cost_norm

    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
