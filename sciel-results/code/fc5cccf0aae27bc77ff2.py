import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a hybrid Best/Worst Fit strategy
    with smooth nonlinear transforms.
    
    Args:
        item: Size of the current item to place
        bins: Array of remaining capacities for each bin
    
    Returns:
        Array of scores for each bin (higher is better)
    """
    # Mask infeasible bins (remaining capacity < item size)
    feasible = bins >= item
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(feasible):
        return scores
    
    remaining = bins[feasible] - item
    
    # Estimate capacity C from the maximum remaining capacity
    # This is a heuristic - actual C might be larger
    C_est = np.max(bins) if len(bins) > 0 else 100.0
    C_est = max(C_est, item)  # Ensure C_est is at least item size
    
    # Relative item size: 0 = very small, 1 = very large (fills entire bin)
    rel_size = item / C_est
    
    # Hybrid strategy:
    # - For small items (rel_size < 0.3): prefer Worst Fit (leave more space)
    # - For large items (rel_size > 0.7): prefer Best Fit (tightest fit)
    # - In between: smooth transition
    
    # Create a weight that blends between Best Fit and Worst Fit
    # weight = 0 means pure Worst Fit, weight = 1 means pure Best Fit
    # Use a smooth sigmoid-like transition
    center = 0.5
    width = 0.3
    weight = 1.0 / (1.0 + np.exp(-(rel_size - center) / width))
    
    # Best Fit score: minimize remaining (prefer tightest fit)
    # Use a smooth power-law: -remaining^alpha where alpha > 1 emphasizes tight fits
    alpha_best = 1.5  # Exponent for Best Fit
    best_score = -remaining**alpha_best
    
    # Worst Fit score: maximize remaining (prefer loosest fit)
    # Normalize remaining to [0, 1] relative to C_est
    remaining_norm = remaining / C_est
    # Use a smooth power-law: remaining_norm^beta where beta < 1 emphasizes loose fits
    alpha_worst = 0.5  # Exponent for Worst Fit
    worst_score = remaining_norm**alpha_worst
    
    # Blend the two scores
    # Both scores should be in a comparable range, so normalize them
    best_min = np.min(best_score)
    best_max = np.max(best_score)
    worst_min = np.min(worst_score)
    worst_max = np.max(worst_score)
    
    # Normalize to [0, 1]
    best_range = best_max - best_min
    worst_range = worst_max - worst_min
    
    if best_range > 1e-10:
        best_norm = (best_score - best_min) / best_range
    else:
        best_norm = np.zeros_like(best_score)
    
    if worst_range > 1e-10:
        worst_norm = (worst_score - worst_min) / worst_range
    else:
        worst_norm = np.zeros_like(worst_score)
    
    # Combine: weight * BestFit + (1-weight) * WorstFit
    combined = weight * best_norm + (1.0 - weight) * worst_norm
    
    scores[feasible] = combined
    
    return scores
