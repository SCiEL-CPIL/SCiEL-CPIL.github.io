"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0  # 5s safety margin

def _vectorized_greedy_fill(cap, weights, values, order, exclude_idx):
    """Vectorized greedy fill: iterate through items in 'order', skip exclude_idx,
    take item if it fits. Returns total value of filled items."""
    sim_cap = cap
    sim_val = 0.0
    w = weights
    v = values
    for j in order:
        if j == exclude_idx:
            continue
        if w[j] <= sim_cap:
            sim_cap -= w[j]
            sim_val += v[j]
    return sim_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        # Fallback: pure density greedy
        weights_f = np.array(remaining_weights, dtype=float)
        values_f = np.array(remaining_values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            dens_f = values_f / weights_f
        idx_f = int(np.argmax(np.where(weights_f <= remaining_capacity, dens_f, -np.inf)))
        return idx_f if weights_f[idx_f] <= remaining_capacity else -1

    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Pre-compute sorting orders for multiple greedy strategies
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    # Handle zero-weight items
    densities = np.where(weights > 0, densities, np.inf)
    
    order_density = np.argsort(-densities)
    order_value = np.argsort(-values)
    order_weight = np.argsort(weights)
    
    # Additional diverse strategies
    # Modified density: value / sqrt(weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        mod_density = values / np.sqrt(np.maximum(weights, 1e-10))
    order_mod_density = np.argsort(-mod_density)
    
    # Value-weighted: value^1.5 / weight
    with np.errstate(divide='ignore', invalid='ignore'):
        val_weighted = np.power(values, 1.5) / np.maximum(weights, 1e-10)
    order_val_weighted = np.argsort(-val_weighted)
    
    all_orders = [order_density, order_value, order_weight, order_mod_density, order_val_weighted]
    
    # UCB-based candidate selection:
    # For each feasible item, compute a UCB score that balances:
    # - Exploitation: value density (how good the item is per unit weight)
    # - Exploration: potential to unlock better combinations (weight synergy)
    # UCB = exploitation + sqrt(ln(n) / n) * exploration_bonus
    
    n_feasible = int(np.sum(feasible_mask))
    ucb_exploration_coeff = np.sqrt(np.log(max(n_feasible, 2)) / max(n_feasible, 2))
    
    # Compute UCB scores for all items
    # Exploitation: normalized density
    dens_normalized = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-10)
    
    # Exploration bonus: items with weight that "fits well" with remaining capacity
    # after selection - items whose weight is a "nice fraction" of remaining capacity
    # tend to leave room for complementary items
    cap_ratio = weights / max(remaining_capacity, 1e-10)
    # Items with cap_ratio near common fractions (1/2, 1/3, 1/4, etc.) may be synergistic
    # Use a smooth measure: distance to nearest "good" ratio
    good_ratios = np.array([0.25, 0.33, 0.5, 0.67, 0.75])
    # For each item, find min distance to good ratios
    dist_to_good = np.min(np.abs(cap_ratio[:, None] - good_ratios[None, :]), axis=1)
    # Exploration bonus: closer to good ratio = higher bonus (inverted)
    exploration_bonus = 1.0 / (1.0 + dist_to_good * 10.0)
    
    ucb_scores = dens_normalized + ucb_exploration_coeff * exploration_bonus
    ucb_scores = np.where(feasible_mask, ucb_scores, -np.inf)
    
    # Select top-K candidates by UCB score for detailed evaluation
    # K should be large enough to capture good options but small enough for speed
    K = min(n, max(20, n // 3))
    ucb_ranked = np.argsort(-ucb_scores)
    candidates = ucb_ranked[:K]
    
    # Also always include top-3 by density and top-3 by value for safety
    top_density = order_density[:3]
    top_value = order_value[:3]
    for idx in list(top_density) + list(top_value):
        if idx not in candidates and feasible_mask[idx]:
            candidates = np.append(candidates, idx)
    
    candidates = np.unique(candidates)
    
    best_val = -1.0
    best_idx = -1
    
    for i in candidates:
        if i >= n or not feasible_mask[i]:
            continue
        if weights[i] > remaining_capacity:
            continue
            
        cap = remaining_capacity - weights[i]
        base_val = values[i]
        
        max_sim_val = 0.0
        
        for order in all_orders:
            sim_val = _vectorized_greedy_fill(cap, weights, values, order, i)
            if sim_val > max_sim_val:
                max_sim_val = sim_val
        
        total_val = base_val + max_sim_val
        
        if total_val > best_val:
            best_val = total_val
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
