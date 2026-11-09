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
    
    # Precompute index arrays for fast lookup
    idx0 = aps[:, 0]
    idx1 = aps[:, 1]
    idx2 = aps[:, 2]
    idx3 = aps[:, 3]
    
    # Precompute pos_to_aps
    pos_to_aps = [[] for _ in range(n)]
    for ap_idx in range(M):
        for j in range(4):
            pos = aps[ap_idx, j]
            pos_to_aps[pos].append(ap_idx)

    def is_valid_ap(colors, ap_idx):
        """Check if the AP at ap_idx is valid (>= 3 distinct colors)."""
        c1 = colors[idx0[ap_idx]]
        c2 = colors[idx1[ap_idx]]
        c3 = colors[idx2[ap_idx]]
        c4 = colors[idx3[ap_idx]]
        
        # Check distinct count
        # If all same: invalid
        if c1 == c2 and c2 == c3 and c3 == c4:
            return False
        # If 3 same, 1 diff: invalid
        if (c1 == c2 and c2 == c3 and c3 != c4) or \
           (c1 == c2 and c2 == c4 and c1 != c3) or \
           (c1 == c3 and c3 == c4 and c1 != c2) or \
           (c2 == c3 and c3 == c4 and c2 != c1):
            return False
        # If 2 same, 2 same: invalid
        if (c1 == c2 and c3 == c4 and c1 != c3) or \
           (c1 == c3 and c2 == c4 and c1 != c2) or \
           (c1 == c4 and c2 == c3 and c1 != c2):
            return False
            
        return True

    def calculate_cost(colors):
        """Calculate total invalid APs."""
        cost = 0
        for ap_idx in range(M):
            if not is_valid_ap(colors, ap_idx):
                cost += 1
        return cost

    def sa(k, max_iters=100000, start_temp=5.0, end_temp=0.01, time_limit=50.0):
        # Random initialization
        current = np.random.randint(0, k, n)
        current_cost = calculate_cost(current)
        
        if current_cost == 0:
            return current

        best = current.copy()
        best_cost = current_cost
        
        T = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
        
        start_time = time.time()
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random position
            pos = np.random.randint(0, n)
            old_color = current[pos]
            
            # Pick a new color
            if k > 1:
                new_color = np.random.randint(0, k - 1)
                if new_color >= old_color:
                    new_color += 1
            else:
                continue
            
            # Incremental update
            delta = 0
            affected_aps = pos_to_aps[pos]
            
            for ap_idx in affected_aps:
                # Check validity before change
                valid_before = is_valid_ap(current, ap_idx)
                
                # Apply change temporarily
                current[pos] = new_color
                
                # Check validity after change
                valid_after = is_valid_ap(current, ap_idx)
                
                # Update cost delta
                if valid_before and not valid_after:
                    delta += 1
                elif not valid_before and valid_after:
                    delta -= 1
            
            # Decide acceptance
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = current.copy()
                    if best_cost == 0:
                        break
            else:
                # Revert change
                current[pos] = old_color
            
            T *= cooling_rate
        
        return best

    # Try increasing number of colors
    start_time = time.time()
    time_limit = 55.0
    
    best_solution = None
    min_colors_used = 128 # Initialize high
    
    for k in range(3, 129):
        if time.time() - start_time > time_limit:
            break
            
        # Run SA multiple times for small k to find robust solution
        # We can run SA until time limit for the first k, or split time.
        # Let's give k=3 a good chance.
        
        runs = 1
        if k == 3:
             # Try a few restarts if fast
             for run in range(3):
                 if time.time() - start_time > time_limit - 5:
                     break
                 solution = sa(k, max_iters=50000, time_limit=time_limit - (time.time() - start_time))
                 cost = calculate_cost(solution)
                 if cost == 0:
                     unique_colors = np.unique(solution)
                     num_colors = len(unique_colors)
                     if num_colors < min_colors_used:
                         min_colors_used = num_colors
                         best_solution = solution.tolist()
                     if min_colors_used == 3:
                         return best_solution
        else:
             solution = sa(k, max_iters=50000, time_limit=time_limit - (time.time() - start_time))
             cost = calculate_cost(solution)
             if cost == 0:
                 unique_colors = np.unique(solution)
                 num_colors = len(unique_colors)
                 if num_colors < min_colors_used:
                     min_colors_used = num_colors
                     best_solution = solution.tolist()
                 # If we found a solution with k colors, and k is small, it's good.
                 # But we want to minimize colors. 
                 # If we found k=4 solution, we should still try to find k=3?
                 # The loop structure tries k=3 first. If it fails, it tries k=4.
                 # So if we reach k=4, k=3 likely failed within the time budget.
                 # We should keep the best solution found so far.
    
    if best_solution is not None:
        return best_solution
        
    # Fallback
    return np.random.randint(0, 128, n).tolist()
