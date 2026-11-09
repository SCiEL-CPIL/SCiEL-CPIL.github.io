import numpy as np
import random
import time

def construct():
    n = 128
    
    # Precompute 4-APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for i in range(1, n - 3 * d + 1):
            i0 = i - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    M = len(aps)
    
    # Precompute for each position, which APs it belongs to and its index in the AP
    pos_to_ap_info = [[] for _ in range(n)]
    for ap_idx in range(M):
        for j in range(4):
            pos = aps[ap_idx, j]
            pos_to_ap_info[pos].append((ap_idx, j))
            
    # Convert to numpy arrays for faster access in delta calc? 
    # Lists of tuples are fine for small counts (~20-30 APs per pos).
    # But converting to arrays might be faster if we do it once.
    pos_to_ap_idxs = [np.array([x[0] for x in info], dtype=np.int32) if info else np.array([], dtype=np.int32) for info in pos_to_ap_info]
    pos_to_ap_js = [np.array([x[1] for x in info], dtype=np.int32) if info else np.array([], dtype=np.int32) for info in pos_to_ap_info]

    def calculate_cost_vectorized(colors):
        cols = colors[aps]
        d1 = cols[:, 0] != cols[:, 1]
        d2 = cols[:, 0] != cols[:, 2]
        d3 = cols[:, 0] != cols[:, 3]
        d4 = cols[:, 1] != cols[:, 2]
        d5 = cols[:, 1] != cols[:, 3]
        d6 = cols[:, 2] != cols[:, 3]
        diffs = d1 + d2 + d3 + d4 + d5 + d6
        return int(np.sum(diffs <= 4))

    def get_cost_delta_vectorized(colors, pos, old_color, new_color):
        ap_idxs = pos_to_ap_idxs[pos]
        js = pos_to_ap_js[pos]
        if len(ap_idxs) == 0:
            return 0
            
        cols = colors[aps[ap_idxs]]
        
        d1 = cols[:, 0] != cols[:, 1]
        d2 = cols[:, 0] != cols[:, 2]
        d3 = cols[:, 0] != cols[:, 3]
        d4 = cols[:, 1] != cols[:, 2]
        d5 = cols[:, 1] != cols[:, 3]
        d6 = cols[:, 2] != cols[:, 3]
        old_diffs = d1 + d2 + d3 + d4 + d5 + d6
        old_invalid_count = np.sum(old_diffs <= 4)
        
        new_cols = cols.copy()
        new_cols[np.arange(len(ap_idxs)), js] = new_color
        
        nd1 = new_cols[:, 0] != new_cols[:, 1]
        nd2 = new_cols[:, 0] != new_cols[:, 2]
        nd3 = new_cols[:, 0] != new_cols[:, 3]
        nd4 = new_cols[:, 1] != new_cols[:, 2]
        nd5 = new_cols[:, 1] != new_cols[:, 3]
        nd6 = new_cols[:, 2] != new_cols[:, 3]
        new_diffs = nd1 + nd2 + nd3 + nd4 + nd5 + nd6
        new_invalid_count = np.sum(new_diffs <= 4)
        
        return int(new_invalid_count - old_invalid_count)

    def sa(k, max_iters=10000, start_temp=10.0, end_temp=0.01):
        best_initial = None
        best_initial_cost = float('inf')
        
        for init_try in range(5):
            current = np.random.randint(0, k, n)
            cost = calculate_cost_vectorized(current)
            if cost < best_initial_cost:
                best_initial_cost = cost
                best_initial = current.copy()
            if cost == 0:
                break
        
        current = best_initial
        current_cost = best_initial_cost
        
        if current_cost == 0:
            return current
        
        best = current.copy()
        best_cost = current_cost
        
        T = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            pos = np.random.randint(0, n)
            old_color = current[pos]
            
            if k > 1:
                new_color = np.random.randint(0, k - 1)
                if new_color >= old_color:
                    new_color += 1
            else:
                continue
            
            delta = get_cost_delta_vectorized(current, pos, old_color, new_color)
            
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                current[pos] = new_color
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = current.copy()
                    if best_cost == 0:
                        break
            
            T *= cooling_rate
        
        return best

    start_time = time.time()
    time_limit = 55.0
    
    for k in range(3, 129):
        if time.time() - start_time > time_limit:
            break
            
        if k == 3:
            for run in range(5):
                if time.time() - start_time > time_limit:
                    break
                solution = sa(k, max_iters=20000, start_temp=5.0, end_temp=0.01)
                if calculate_cost_vectorized(solution) == 0:
                    return solution.tolist()
        else:
            solution = sa(k, max_iters=5000, start_temp=5.0, end_temp=0.01)
            if calculate_cost_vectorized(solution) == 0:
                return solution.tolist()
        
    return np.random.randint(0, 3, n).tolist()
