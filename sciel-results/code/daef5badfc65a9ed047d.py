"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # prizes: (5, 100), weights: (5, 100)
    n_constraints, n_items = prizes.shape
    
    eps = 1e-9
    
    # Per-constraint prize and weight
    # prizes[i, j] = prize of item j in constraint i
    # weights[i, j] = weight of item j in constraint i
    
    # Total prize per item: sum across all prize dimensions
    total_prize = np.sum(prizes, axis=0)  # shape (100,)
    
    # Per-constraint efficiency: prize[i,j] / weight[i,j]
    # Safe division
    safe_weights = np.maximum(weights, eps)
    eff = prizes / safe_weights  # shape (5, 100)
    
    # Soft-min of per-constraint efficiency ratios
    # This approximates the bottleneck constraint while being smooth
    # Use a large temperature for soft-min to emphasize the minimum
    alpha = 20.0  # larger alpha -> closer to true min
    # Soft-min: -1/alpha * log(sum(exp(-alpha * eff)))
    neg_alpha_eff = -alpha * eff
    # Subtract max for numerical stability
    max_neg = np.max(neg_alpha_eff, axis=0, keepdims=True)
    exp_neg = np.exp(neg_alpha_eff - max_neg)
    sum_exp = np.sum(exp_neg, axis=0)
    soft_min = -(1.0 / alpha) * (np.log(sum_exp) + max_neg.squeeze(0))
    
    # Normalize soft_min to [0, 1]
    sm_min = np.min(soft_min)
    sm_max = np.max(soft_min)
    soft_min_norm = (soft_min - sm_min) / (sm_max - sm_min + eps)
    
    # Also compute prize-based score
    prize_norm = (total_prize - np.min(total_prize)) / (np.max(total_prize) - np.min(total_prize) + eps)
    
    # Combine: soft-min efficiency is primary, prize is secondary
    heuristic_scores = 0.7 * soft_min_norm + 0.3 * prize_norm
    
    # Ensure strictly positive
    heuristic_scores = np.maximum(heuristic_scores, eps)
    
    # Handle non-finite values
    heuristic_scores = np.where(np.isfinite(heuristic_scores), heuristic_scores, eps)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return heuristic_scores
# EVOLVE-BLOCK-END
