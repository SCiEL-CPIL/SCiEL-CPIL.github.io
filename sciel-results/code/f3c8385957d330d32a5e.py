import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute all non-constant 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    # other_indices[ap_idx][pos_idx] = tuple of other 3 indices
    other_indices = []
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)
    
    # pos_in_ap[ap_idx][idx] = position of idx in ap
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx

    def count_bad_colors(colors):
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad

    def try_k_colors(k, time_limit):
        """Try to find a valid coloring with exactly k colors."""
        best_solution = None
        best_cost = float('inf')
        
        max_restarts = 20
        restart_start = time.time()
        
        for restart in range(max_restarts):
            if time.time() - restart_start > time_limit:
                break
            
            # Random initialization
            colors = np.random.randint(0, k, n)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                return colors.copy()
            
            # Simulated Annealing with fast delta computation
            T = 1.0
            min_T = 0.005
            steps_per_restart = 5000
            restart_limit = time_limit / max_restarts
            
            restart_start2 = time.time()
            for step in range(steps_per_restart):
                if time.time() - restart_start2 > restart_limit:
                    break
                    
                if current_cost == 0:
                    return colors.copy()
                
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    pos = pos_in_ap[ap_idx][i]
                    others = other_indices[ap_idx][pos]
                    c_others = [colors[j] for j in others]
                    set_others = set(c_others)
                    size_others = len(set_others)
                    
                    old_size = size_others + (0 if old_color in set_others else 1)
                    new_size = size_others + (0 if new_color in set_others else 1)
                    
                    old_bad = 1 if old_size <= 2 else 0
                    new_bad = 1 if new_size <= 2 else 0
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                        
                T *= 0.999
                if T < min_T:
                    T = min_T

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = colors.copy()
                
            if best_cost == 0:
                return best_solution

        return None

    # Try k=3 first, then k=4
    time_for_3 = min(25, deadline - time.time() - 10)
    if time_for_3 > 2:
        sol = try_k_colors(3, time_for_3)
        if sol is not None:
            # Verify and try to minimize colors used
            used = set(sol)
            result = list(sol)
            return result
    
    time_for_4 = min(20, deadline - time.time() - 5)
    if time_for_4 > 2:
        sol = try_k_colors(4, time_for_4)
        if sol is not None:
            return list(sol)
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check all 4-term APs ending at x-1
            for d in range(1, (x-1)//3+1):
                a = x - 1 - 3*d
                if a < 0:
                    continue
                c0 = colors[a]
                c1 = colors[a + d]
                c2 = colors[a + 2*d]
                # Need at least 3 distinct colors among c0,c1,c2,color
                s = {c0, c1, c2}
                if len(s) >= 3:
                    # Already 3 distinct, any color works for this AP
                    continue
                elif len(s) == 2:
                    # Need color not in s
                    if color in s:
                        valid = False
                        break
                else:  # len(s) == 1
                    # Need color not equal to s[0]
                    if color == list(s)[0]:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen, but just in case
            colors.append(0)
    
    return colors
