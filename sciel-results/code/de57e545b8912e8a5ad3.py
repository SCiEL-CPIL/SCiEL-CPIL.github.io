import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of multiple-constraint knapsack.
    
    Strategy: Multi-criteria ranking by normalizing each constraint's weight independently
    to [0,1] and computing the heuristic as the product of normalized prize and the inverse
    of the maximum normalized weight across all constraints. This prioritizes items that
    are highly valuable relative to their worst-case resource consumption.
    
    Additionally, we incorporate a soft-min of per-constraint efficiency ratios to further
    guide ants toward items that are efficient across all dimensions.
    """
    n_constraints = prizes.shape[0]  # 5
    n_items = prizes.shape[1]        # 100
    
    # Safe weights to avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    
    # Per-constraint efficiency: prize / weight
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # Normalize each constraint's weight independently to [0, 1]
    # weight_norm[c, i] = (weights[c, i] - min_c) / (max_c - min_c + eps)
    weight_min = weights.min(axis=1, keepdims=True)  # shape (5, 1)
    weight_max = weights.max(axis=1, keepdims=True)  # shape (5, 1)
    weight_range = weight_max - weight_min + 1e-9
    weight_norm = (weights - weight_min) / weight_range  # shape (5, 100), in [0, 1]
    
    # Maximum normalized weight across all constraints for each item
    # This represents the worst-case resource consumption in normalized terms
    max_norm_weight = weight_norm.max(axis=0)  # shape (100,)
    
    # Inverse of max normalized weight (add small epsilon to avoid division by zero)
    inv_max_norm_weight = 1.0 / (max_norm_weight + 1e-9)  # shape (100,)
    
    # Normalize total prize to [0, 1]
    total_prize = prizes.sum(axis=0)  # shape (100,)
    prize_min = total_prize.min()
    prize_max = total_prize.max()
    prize_range = prize_max - prize_min + 1e-9
    prize_norm = (total_prize - prize_min) / prize_range  # shape (100,), in [0, 1]
    
    # Primary heuristic: product of normalized prize and inverse of max normalized weight
    primary_heuristic = prize_norm * inv_max_norm_weight  # shape (100,)
    
    # Soft-min of per-constraint efficiency ratios
    # soft-min via log-sum-exp: -log(sum(exp(-x))) approximates min(x)
    # Use a moderate temperature for smoothness
    beta = 10.0  # inverse temperature; larger -> closer to hard min
    neg_eff = -beta * efficiency  # shape (5, 100)
    m = neg_eff.max(axis=0, keepdims=True)
    log_sum = np.log(np.sum(np.exp(neg_eff - m), axis=0)) + m[0]
    soft_min_efficiency = -log_sum / beta  # shape (100,)
    
    # Normalize soft-min efficiency to [0, 1]
    sme_min = soft_min_efficiency.min()
    sme_max = soft_min_efficiency.max()
    sme_range = sme_max - sme_min + 1e-9
    soft_min_norm = (soft_min_efficiency - sme_min) / sme_range  # shape (100,), in [0, 1]
    
    # Combine primary heuristic with soft-min efficiency
    # Weight the primary heuristic more heavily since it directly models the strategy
    heuristic = 0.7 * primary_heuristic + 0.3 * soft_min_norm
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
