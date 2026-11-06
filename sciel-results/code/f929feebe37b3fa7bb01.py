import numpy as np
import time
import random

def construct():
    N = 128
    aps = []
    ap_pos_indices = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            ap_idx = len(aps)
            aps.append((p1, p2, p3, p4))
            ap_pos_indices[p1].append(ap_idx)
            ap_pos_indices[p2].append(ap_idx)
            ap_pos_indices[p3].append(ap_idx)
            ap_pos_indices[p4].append(ap_idx)
            
    num_aps = len(aps)
    
    def get_ap_cost(ap_idx, colors):
        p1, p2, p3, p4 = aps[ap_idx]
        c1, c2, c3, c4 = colors[p1], colors[p2], colors[p3], colors[p4]
        distinct = len({c1, c2, c3, c4})
        return max(0, 3 - distinct)

    def try_coloring(k_colors, time_limit=50):
        colors = np.random.randint(0, k_colors, N)
        ap_costs = np.array([get_ap_cost(i, colors) for i in range(num_aps)])
        total_cost = np.sum(ap_costs)
        
        if total_cost == 0:
            return colors, 0
        
        start_time = time.time()
        T = 5.0
        cooling = 0.9995
        min_T = 0.001
        
        best_colors = colors.copy()
        best_cost = total_cost
        
        while T > min_T and (time.time() - start_time) < time_limit:
            x = np.random.randint(0, N)
            old_color = colors[x]
            
            # Choose new color
            # Uniform random from k_colors - 1
            new_color = np.random.randint(0, k_colors)
            if new_color == old_color:
                new_color = (new_color + 1) % k_colors
                
            affected_aps = ap_pos_indices[x]
            delta_cost = 0
            
            # Save old costs
            old_costs = {}
            for ap_idx in affected_aps:
                old_costs[ap_idx] = ap_costs[ap_idx]
                
            colors[x] = new_color
            
            for ap_idx in affected_aps:
                new_cost = get_ap_cost(ap_idx, colors)
                delta_cost += new_cost - old_costs[ap_idx]
                ap_costs[ap_idx] = new_cost
                
            if delta_cost <= 0 or np.random.rand() < np.exp(-delta_cost / T):
                total_cost += delta_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors, 0
            else:
                colors[x] = old_color
                # Revert ap_costs
                for ap_idx in affected_aps:
                    ap_costs[ap_idx] = old_costs[ap_idx]
                
            T *= cooling
            
        return best_colors, best_cost

    # Try 3 colors
    colors_3, cost_3 = try_coloring(3, time_limit=50)
    if cost_3 == 0:
        return [int(c) for c in colors_3]
        
    # Try 4 colors
    colors_4, cost_4 = try_coloring(4, time_limit=8)
    if cost_4 == 0:
        return [int(c) for c in colors_4]
        
    # If neither is perfect, return the one with lower cost?
    # But invalid results score 0.
    # We should try to ensure validity.
    # If we can't find valid, we are stuck.
    # But for N=128, 3-coloring should be findable.
    
    # Fallback
    if cost_3 < cost_4:
        return [int(c) for c in colors_3]
    else:
        return [int(c) for c in colors_4]
