import numpy as np
import time

_DEADLINE = 330.0
_START_TIME = time.time()

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Uses a Scarcity-Weighted Geometric Mean of per-constraint prize-to-weight ratios.
    1. The geometric mean ensures that an item is only highly rated if it is efficient 
       across all constraints, avoiding the 'bottleneck' problem where an item is 
       efficient in most constraints but terrible in one.
    2. The weights for the geometric mean are determined by the total weight of all 
       items in each constraint. Constraints that are 'tighter' (higher total demand 
       relative to capacity) are given more weight in the heuristic, penalizing items 
       that are inefficient in those specific tight constraints more heavily.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to simple geometric mean if running out of time
        p = np.clip(prizes, 1e-9, None)
        w = np.clip(weights, 1e-9, None)
        ratios = p / w
        log_ratios = np.log(ratios)
        mean_log_ratio = np.mean(log_ratios, axis=0)
        heuristic = np.exp(mean_log_ratio)
        heuristic = np.maximum(heuristic, 1e-9)
        mean_h = np.mean(heuristic)
        if mean_h > 0:
            heuristic = heuristic / mean_h
        return heuristic

    # Ensure inputs are float64 arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    
    # 1. Calculate per-constraint ratios: Prize / Weight
    # Clip weights to avoid division by zero or log(0)
    safe_weights = np.clip(weights, eps, None)
    safe_prizes = np.clip(prizes, eps, None)
    
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    
    # 2. Calculate scarcity weights using top-k/mean ratio for binding constraint identification
    capacity = 25.0
    total_weights_per_constraint = np.sum(weights, axis=1)  # Shape (5,)
    tightness = total_weights_per_constraint / capacity  # Shape (5,)
    
    # For each constraint, compute the ratio of top-k weights to mean weight
    # This identifies constraints where a few items dominate the weight (more binding)
    n_items = weights.shape[1]
    k = max(1, n_items // 5)  # Top 20% of items
    
    top_k_weights = np.sort(weights, axis=1)[:, -k:]  # Shape (5, k)
    mean_weights = np.mean(weights, axis=1)  # Shape (5,)
    
    # Scarcity ratio: how much more concentrated the weights are compared to uniform
    # If top-k items have much higher weight than average, the constraint is more binding
    top_k_sum = np.sum(top_k_weights, axis=1)  # Shape (5,)
    mean_total = mean_weights * n_items  # Shape (5,)
    
    # Ratio of top-k sum to what it would be if uniformly distributed
    uniform_top_k_sum = mean_weights * k
    scarcity_ratio = top_k_sum / np.maximum(uniform_top_k_sum, eps)  # Shape (5,)
    
    # Combine with overall tightness
    # Constraints that are both tight overall AND have concentrated weights are most binding
    scarcity_weights = tightness * scarcity_ratio
    sum_scarcity = np.sum(scarcity_weights)
    
    if sum_scarcity < eps:
        # Fallback if all weights are zero (unlikely given uniform [0,1])
        scarcity_weights = np.ones(5)
        sum_scarcity = 5.0
    
    # 3. Compute Weighted Geometric Mean with soft-min bottleneck penalty
    log_ratios = np.log(ratios)  # Shape (5, 100)
    
    # Weighted sum of log ratios
    # scarcity_weights shape (5,), log_ratios shape (5, 100)
    # Use broadcasting: scarcity_weights[:, np.newaxis] * log_ratios
    weighted_log_sum = np.sum((scarcity_weights[:, np.newaxis] / sum_scarcity) * log_ratios, axis=0)
    
    # Add soft-min bottleneck penalty using log-sum-exp
    # This smoothly penalizes items that are very inefficient in any constraint
    # The soft-min approximates min_c(ratio_c) using log-sum-exp
    # log(soft_min) = -log(sum(exp(-log_ratios)))
    # We use a temperature parameter to control the smoothness
    temperature = 5.0
    neg_log_ratios = -log_ratios / temperature  # Shape (5, 100)
    max_neg = np.max(neg_log_ratios, axis=0, keepdims=True)  # Shape (1, 100)
    shifted = neg_log_ratios - max_neg  # Shape (5, 100)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=0)) + max_neg[:, 0]  # Shape (100,)
    soft_min_log = -log_sum_exp  # This is log(soft_min(ratios))
    
    # Blend the weighted geometric mean with the soft-min bottleneck
    # The soft-min ensures items that are terrible in any constraint are heavily penalized
    blend_alpha = 0.3  # Weight of soft-min penalty
    combined_log = (1 - blend_alpha) * weighted_log_sum + blend_alpha * soft_min_log
    
    # Convert back to linear scale
    heuristic = np.exp(combined_log)
    
    # 4. Ensure positivity and finiteness
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 5. Normalize to have mean 1
    # This stabilizes the interaction with pheromones which start at 1.0.
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
