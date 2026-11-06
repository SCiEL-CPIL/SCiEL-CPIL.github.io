import numpy as np
import time
import random

def construct():
    n = 128
    # Precompute all 4-term APs
    aps = []
    aps_by_pos = [[] for _ in range(n)]
    ap_idx = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            for idx in [i0, i1, i2, i3]:
                aps_by_pos[idx].append(ap_idx)
            ap_idx += 1
            
    num_aps = len(aps)
    
    def is_violated(ap, colors):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        # Check if < 3 distinct colors
        if c0 == c1 == c2 == c3:
            return True
        if len({c0, c1, c2, c3}) < 3:
            return True
        return False

    def optimize(num_colors, time_limit=5.0):
        start_time = time.time()
        colors = np.random.randint(0, num_colors, n)
        
        # Initialize violation array and cost
        violated = np.zeros(num_aps, dtype=bool)
        cost = 0
        for k, ap in enumerate(aps):
            if is_violated(ap, colors):
                violated[k] = True
                cost += 1
                
        if cost == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_cost = cost
        
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        max_iters = 20000
        
        iter_count = 0
        while T > min_T and iter_count < max_iters and time.time() - start_time < time_limit:
            pos = np.random.randint(0, n)
            old_color = colors[pos]
            
            # Find best new color
            best_new_color = old_color
            best_new_cost = cost
            
            for new_color in range(num_colors):
                if new_color == old_color:
                    continue
                
                # Calculate delta cost
                colors[pos] = new_color
                new_cost = cost
                for ap_idx in aps_by_pos[pos]:
                    ap = aps[ap_idx]
                    was_violated = violated[ap_idx]
                    is_now_violated = is_violated(ap, colors)
                    if was_violated and not is_now_violated:
                        new_cost -= 1
                    elif not was_violated and is_now_violated:
                        new_cost += 1
                colors[pos] = old_color # Revert for next trial
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_color = new_color
            
            # Apply best move
            if best_new_color != old_color:
                colors[pos] = best_new_color
                for ap_idx in aps_by_pos[pos]:
                    ap = aps[ap_idx]
                    is_now_violated = is_violated(ap, colors)
                    if violated[ap_idx] and not is_now_violated:
                        violated[ap_idx] = False
                    elif not violated[ap_idx] and is_now_violated:
                        violated[ap_idx] = True
                cost = best_new_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors.copy()
                    
                if best_cost == 0:
                    return best_colors.tolist()
                
            T *= cooling_rate
            iter_count += 1
            
        if best_cost == 0:
            return best_colors.tolist()
        return None

    # Try 3 colors
    result_3 = optimize(3, time_limit=10.0)
    if result_3 is not None:
        return result_3
        
    # Try 4 colors
    result_4 = optimize(4, time_limit=20.0)
    if result_4 is not None:
        return result_4
        
    # Fallback: Random restarts for 4 colors
    for _ in range(50):
        result_4 = optimize(4, time_limit=1.0)
        if result_4 is not None:
            return result_4
            
    # Last resort: Return a valid 4-coloring found by a simple greedy?
    # If all else fails, return a pattern that is likely valid or just raise?
    # We must return a list.
    # Let's return the best_colors from the last run, even if invalid? 
    # No, "invalid result ... scores 0".
    # We should ensure we find a valid one.
    # 4 colors should be easy.
    
    return result_4 if result_4 is not None else list(range(128))
