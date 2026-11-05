import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0

def _fractional_bound(weights, values, capacity):
    if capacity <= 0:
        return 0.0
    total_value = 0.0
    remaining_capacity = capacity
    for w, v in zip(weights, values):
        if remaining_capacity <= 0:
            break
        if w <= remaining_capacity:
            total_value += v
            remaining_capacity -= w
        else:
            if w > 0:
                total_value += (remaining_capacity / w) * v
            break
    return total_value

def _solve_knapsack_bnb(weights, values, capacity):
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    densities = np.zeros(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    densities[weights == 0] = np.inf
    
    order = np.argsort(-densities, kind='stable')
    sorted_w = weights[order]
    sorted_v = values[order]
    
    best_value = 0.0
    
    def dfs(idx, curr_val, curr_w):
        nonlocal best_value
        
        if idx == n:
            if curr_val > best_value:
                best_value = curr_val
            return
        
        rem_cap = capacity - curr_w
        if rem_cap < 0:
            return
            
        # Bound calculation
        bound = curr_val
        temp_cap = rem_cap
        for k in range(idx, n):
            if sorted_w[k] <= temp_cap:
                bound += sorted_v[k]
                temp_cap -= sorted_w[k]
            else:
                if sorted_w[k] > 0:
                    bound += (temp_cap / sorted_w[k]) * sorted_v[k]
                break
                
        if bound <= best_value:
            return
            
        # Exclude
        dfs(idx + 1, curr_val, curr_w)
        
        # Include
        if curr_w + sorted_w[idx] <= capacity:
            dfs(idx + 1, curr_val + sorted_v[idx], curr_w + sorted_w[idx])
            
    dfs(0, 0.0, 0.0)
    return best_value

def _solve_knapsack_dp(weights, values, capacity):
    C_int = int(capacity)
    if C_int < 0:
        return 0.0
    n = len(weights)
    dp = np.zeros(C_int + 1)
    for i in range(n):
        wi = int(weights[i])
        vi = values[i]
        if wi > C_int:
            continue
        for c in range(C_int, wi - 1, -1):
            new_val = dp[c - wi] + vi
            if new_val > dp[c]:
                dp[c] = new_val
    return dp[C_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    if time.time() - _start_time > _TIMEOUT:
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights)
    v = np.array(remaining_values)
    C = remaining_capacity
    
    feasible_indices = np.where(w <= C)[0]
    if len(feasible_indices) == 0:
        return -1
        
    # Check if weights are integers
    is_int = np.allclose(w, np.round(w))
    
    best_total_value = -np.inf
    best_idx = -1
    
    for i in feasible_indices:
        # Time check inside loop
        if time.time() - _start_time > _TIMEOUT - 5.0:
             # Fall back to greedy for this item or break
             # Just break and use best so far, or simple greedy
             break
             
        w_i = w[i]
        v_i = v[i]
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rem_w = w[mask]
        rem_v = v[mask]
        
        rem_cap = C - w_i
        
        if is_int:
            sub_val = _solve_knapsack_dp(rem_w, rem_v, rem_cap)
        else:
            sub_val = _solve_knapsack_bnb(rem_w, rem_v, rem_cap)
            
        total_val = v_i + sub_val
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_idx = i
            
    if best_idx == -1:
        # Fallback if loop broke early or no item found
        densities = v[feasible_indices] / w[feasible_indices]
        best_local = np.argmax(densities)
        return int(feasible_indices[best_local])
        
    return int(best_idx)
