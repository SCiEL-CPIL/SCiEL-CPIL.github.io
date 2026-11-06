import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin out of 150s total
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)
    
    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)
    
    if n_f == 0:
        return -1
        
    # Compute densities
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = np.where(v_f[~non_zero_w] > 0, np.inf, 0.0)
    
    # Sort by density descending
    sort_idx = np.argsort(-d_f)
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    d_s = d_f[sort_idx]
    
    # Prefix sums for LP bound
    pw = np.cumsum(w_s)
    pv = np.cumsum(v_s)
    
    def get_lp_value(cap):
        if cap <= 1e-9:
            return 0.0
        k = np.searchsorted(pw, cap, side='right') - 1
        if k < 0:
            if len(w_s) == 0:
                return 0.0
            return cap * d_s[0]
        val = pv[k]
        remaining_cap = cap - pw[k]
        if k < len(w_s) - 1:
            next_w = w_s[k + 1]
            next_d = d_s[k + 1]
            val += remaining_cap * next_d
        return val
    
    # Try multiple heuristic strategies and pick the best first item
    # Strategy 1: Greedy by density
    # Strategy 2: Greedy by value
    # Strategy 3: Greedy by value/weight with lookahead
    # Strategy 4: Dynamic programming approximation
    
    best_score = -1.0
    best_i_local = -1
    
    # Helper to simulate a greedy packing and compute total value
    def simulate_greedy(capacity, weights, values, order):
        total_value = 0.0
        current_cap = capacity
        for idx in order:
            if weights[idx] > current_cap:
                continue
            total_value += values[idx]
            current_cap -= weights[idx]
            if current_cap <= 1e-9:
                break
        return total_value
    
    # Strategy 1: Density order
    density_order = sort_idx
    if time.time() < deadline:
        score1 = simulate_greedy(C, w_f, v_f, density_order)
        if score1 > best_score:
            best_score = score1
            best_i_local = density_order[0]
    
    # Strategy 2: Value order
    if time.time() < deadline:
        value_order = np.argsort(-v_f)
        score2 = simulate_greedy(C, w_f, v_f, value_order)
        if score2 > best_score:
            best_score = score2
            best_i_local = value_order[0]
    
    # Strategy 3: Weight order (lightest first)
    if time.time() < deadline:
        weight_order = np.argsort(w_f)
        score3 = simulate_greedy(C, w_f, v_f, weight_order)
        if score3 > best_score:
            best_score = score3
            best_i_local = weight_order[0]
    
    # Strategy 4: 2-step lookahead with LP bound
    if time.time() < deadline:
        best_score_2 = -1.0
        best_i_2 = -1
        for i in range(n_f):
            if w_f[i] > C:
                continue
            rem_cap = C - w_f[i]
            # Find best second item
            best_j_score = -1.0
            best_j = -1
            for j in range(n_f):
                if i == j:
                    continue
                if w_f[i] + w_f[j] > C:
                    continue
                rem_cap2 = rem_cap - w_f[j]
                score = v_f[i] + v_f[j] + get_lp_value(rem_cap2)
                if score > best_j_score:
                    best_j_score = score
                    best_j = j
            if best_j != -1:
                total_score = v_f[i] + v_f[best_j] + get_lp_value(C - w_f[i] - w_f[best_j])
            else:
                total_score = v_f[i] + get_lp_value(rem_cap)
            if total_score > best_score_2:
                best_score_2 = total_score
                best_i_2 = i
        if best_i_2 != -1 and best_score_2 > best_score:
            best_score = best_score_2
            best_i_local = best_i_2
    
    # Strategy 5: DP approximation with value scaling
    if time.time() < deadline and n_f <= 200:
        # Scale values to make DP tractable
        max_val = float(np.max(v_f))
        if max_val > 0:
            # Target DP table size ~ 10000
            scale = max(1.0, max_val / 10000.0)
            v_scaled = np.round(v_f / scale).astype(int)
            max_scaled_val = int(np.max(v_scaled))
            
            if max_scaled_val <= 20000:
                # DP: dp[cap] = max value achievable with capacity cap
                # Use integer capacity scaling
                max_w = float(np.max(w_f))
                if max_w > 0:
                    w_scale = max(1.0, max_w / 10000.0)
                    w_int = np.round(w_f / w_scale).astype(int)
                    max_w_int = int(np.max(w_int))
                    
                    if max_w_int <= 10000:
                        dp = np.zeros(max_w_int + 1, dtype=int)
                        # Track which item achieves each dp value
                        # For each item, update dp in reverse
                        item_used = np.zeros((max_w_int + 1, n_f), dtype=int)
                        
                        for i in range(n_f):
                            wi = w_int[i]
                            vi = v_scaled[i]
                            if wi <= 0:
                                continue
                            for cap in range(max_w_int, wi - 1, -1):
                                new_val = dp[cap - wi] + vi
                                if new_val > dp[cap]:
                                    dp[cap] = new_val
                                    item_used[cap, i] = 1
                                    # We need to track the path, but for now just find first item
                                    # that leads to optimal solution
                            
                        # Find the first item on the optimal path
                        # Start from max capacity, trace back
                        optimal_cap = int(C / w_scale)
                        optimal_cap = min(optimal_cap, max_w_int)
                        
                        # Reconstruct: find which item was used to achieve dp[optimal_cap]
                        # This is complex, so instead let's just use the DP to score each candidate
                        # For each candidate item i, compute the best value if we take i first
                        best_dp_score = -1.0
                        best_dp_i = -1
                        for i in range(n_f):
                            wi = w_int[i]
                            vi = v_scaled[i]
                            if wi > optimal_cap:
                                continue
                            rem_cap_int = optimal_cap - wi
                            # Best value from remaining items with capacity rem_cap_int
                            # This requires a DP that excludes item i, which is expensive
                            # Instead, approximate: dp[rem_cap_int] is an upper bound
                            score = vi + dp[rem_cap_int]
                            if score > best_dp_score:
                                best_dp_score = score
                                best_dp_i = i
                        
                        if best_dp_i != -1:
                            # Convert back to real value for comparison
                            real_score = best_dp_score * scale
                            if real_score > best_score:
                                best_score = real_score
                                best_i_local = best_dp_i
    
    if best_i_local == -1:
        return -1
    
    return int(feasible_indices[best_i_local])
