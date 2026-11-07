"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d).

Uses a Pareto-frontier selection strategy over (predicted improvement,
uncertainty, cost) to balance exploration and cost without arbitrary
weight tuning.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Compute raw expected improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    n = len(mu)
    
    # Objectives to maximize: ei, sigma, -cost
    obj1 = ei
    obj2 = sigma
    obj3 = -cost
    
    # Normalize objectives to [0, 1] to make distance comparable
    def normalize(v):
        v_min = np.min(v)
        v_max = np.max(v)
        if v_max - v_min < 1e-12:
            return np.ones_like(v)
        return (v - v_min) / (v_max - v_min)
        
    n_obj1 = normalize(obj1)
    n_obj2 = normalize(obj2)
    n_obj3 = normalize(obj3)
    
    # Ideal point is (1, 1, 1) in normalized space
    # Distance to ideal point
    dist = np.sqrt((1.0 - n_obj1)**2 + (1.0 - n_obj2)**2 + (1.0 - n_obj3)**2)
    
    # Identify Pareto Frontier
    on_frontier = np.ones(n, dtype=bool)
    
    block_size = 100
    sub_block_size = 50
    
    # Criteria matrix for domination check (maximization)
    criteria = np.column_stack([obj1, obj2, obj3])
    
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        for s_start in range(start, end, sub_block_size):
            s_end = min(s_start + sub_block_size, end)
            cand = criteria[s_start:s_end]  # shape [c, 3]
            
            for i_start in range(0, n, 200):
                i_end = min(i_start + 200, n)
                other = criteria[i_start:i_end]  # shape [m, 3]
                
                diff = other[:, np.newaxis, :] - cand[np.newaxis, :, :]
                all_ge = np.all(diff >= 0, axis=2)
                any_gt = np.any(diff > 0, axis=2)
                dominates = all_ge & any_gt
                
                is_dominated_in_block = np.any(dominates, axis=0)
                if np.any(is_dominated_in_block):
                    local_indices = np.where(is_dominated_in_block)[0]
                    global_indices = s_start + local_indices
                    on_frontier[global_indices] = False

    # Assign scores
    # Frontier points: score based on distance to ideal (closer is better, so negative distance)
    # Non-frontier points: very low score
    
    max_dist = np.max(dist)
    # Base score for non-frontier is very low
    scores = np.full(n, -10.0 * max_dist)
    
    # For frontier points, use -dist + small bonus for EI to break ties
    # The bonus ensures that among points with similar distance, higher EI is preferred
    scores[on_frontier] = -dist[on_frontier] + 1e-6 * n_obj1[on_frontier]
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
