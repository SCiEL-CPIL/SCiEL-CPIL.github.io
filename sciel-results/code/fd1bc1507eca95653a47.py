import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _dp_optimal_value(capacity, weights, values):
    """Compute optimal value using DP for small capacity problems."""
    if capacity <= 0 or len(weights) == 0:
        return 0
    # Use integer scaling for capacity to make DP feasible
    scale = 100
    cap_int = int(capacity * scale)
    w_int = np.array([int(w * scale) for w in weights])
    v_int = np.array([int(v * scale) for v in values])
    
    # Filter to only feasible items
    feasible = w_int <= cap_int
    if not np.any(feasible):
        return 0
    w_f = w_int[feasible]
    v_f = v_int[feasible]
    
    # DP table
    dp = np.zeros(cap_int + 1, dtype=np.int64)
    for i in range(len(w_f)):
        wi = w_f[i]
        vi = v_f[i]
        if wi <= 0:
            dp[:] += vi
        else:
            for c in range(cap_int, wi - 1, -1):
                val = dp[c - wi] + vi
                if val > dp[c]:
                    dp[c] = val
    return dp[cap_int] / scale

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check for zero weight items with positive value
    zero_weight_mask = (remaining_weights == 0) & (remaining_values > 0)
    if np.any(zero_weight_mask):
        return int(np.where(zero_weight_mask)[0][0])

    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    m = len(feasible_indices)
    
    # Calculate densities
    densities = feasible_values / feasible_weights
    
    # Sort indices by density descending
    sorted_order = np.argsort(-densities)
    
    # Build candidate pool: top by density + top by value
    top_k = min(m, max(10, m // 5))
    top_density = set(sorted_order[:top_k].tolist())
    top_value_order = np.argsort(-feasible_values)
    top_value = set(top_value_order[:top_k].tolist())
    
    # Union of candidates
    candidate_set = top_density | top_value
    if m <= 50:
        candidate_set = set(range(m))
    
    candidates = sorted(candidate_set)
    
    # Pre-extract sorted arrays for faster access
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    
    best_score = -1
    best_idx = -1
    start_time = time.time()
    deadline = start_time + 0.5  # Safety margin per call
    
    for k in candidates:
        if time.time() > deadline:
            break
            
        idx_in_feasible = feasible_indices[k]
        w = feasible_weights[k]
        v = feasible_values[k]
        
        # Simulate picking this item
        rem_cap = remaining_capacity - w
        sim_val = v
        
        # Use DP for accurate evaluation if remaining capacity is small enough
        if rem_cap <= 15.0:
            # Build arrays of remaining feasible items (excluding k)
            remaining_w_list = []
            remaining_v_list = []
            for j in range(m):
                if j == k:
                    continue
                if feasible_weights[j] <= rem_cap:
                    remaining_w_list.append(feasible_weights[j])
                    remaining_v_list.append(feasible_values[j])
            
            if remaining_w_list:
                dp_val = _dp_optimal_value(rem_cap, np.array(remaining_w_list), np.array(remaining_v_list))
                sim_val = v + dp_val
        
        # Greedy fill the rest using pre-sorted arrays as fallback
        else:
            for j in range(m):
                if j == k:
                    continue
                w_j = sorted_weights[j]
                if w_j <= rem_cap:
                    sim_val += sorted_values[j]
                    rem_cap -= w_j
                    if rem_cap <= 0:
                        break
        
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx_in_feasible

    if best_idx == -1:
        best_idx = feasible_indices[sorted_order[0]]
        
    return int(best_idx)
