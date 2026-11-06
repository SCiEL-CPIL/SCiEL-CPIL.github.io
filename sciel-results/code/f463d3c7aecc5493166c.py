import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    n = 128
    
    # Precompute all 4-term arithmetic progressions
    # Indices are 0-based: (a, a+d, a+2d, a+3d)
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    if not ap_list:
        return list(range(n))
        
    aps = np.array(ap_list, dtype=np.int32)
    
    # Helper to count violations
    # Violation if number of equal pairs in the 4-tuple is >= 2
    def count_bad(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        # Count equal pairs
        eq01 = (c0 == c1)
        eq02 = (c0 == c2)
        eq03 = (c0 == c3)
        eq12 = (c1 == c2)
        eq13 = (c1 == c3)
        eq23 = (c2 == c3)
        
        num_eq = eq01 + eq02 + eq03 + eq12 + eq13 + eq23
        bad_mask = num_eq >= 2
        return np.sum(bad_mask)
    
    # Try to find a valid coloring with k colors
    def search_k_colors(k, time_limit):
        end_time = time.time() + time_limit
        random.seed(42) # Reproducibility
        
        # Try multiple restarts
        while time.time() < end_time:
            # Initialize random coloring
            colors = np.array([random.randint(0, k - 1) for _ in range(n)], dtype=np.int32)
            current_bad = count_bad(colors)
            
            if current_bad == 0:
                return colors
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.995
            no_improve_count = 0
            best_bad = current_bad
            best_colors = colors.copy()
            
            while time.time() < end_time and T > 0.01:
                # Pick a random position
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                
                # Pick a new color
                if k > 1:
                    new_color = random.randint(0, k - 1)
                    if new_color == old_color:
                        continue
                else:
                    continue
                
                colors[pos] = new_color
                new_bad = count_bad(colors)
                
                delta = new_bad - current_bad
                
                if delta < 0 or (random.random() < np.exp(-delta / T)):
                    current_bad = new_bad
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_colors = colors.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    colors[pos] = old_color
                    no_improve_count += 1
                
                T *= cooling
                
                # Restart if stuck for too long
                if no_improve_count > 1000:
                    break
            
            # If we found a solution in this restart, return it
            if best_bad == 0:
                return best_colors
        
        return None

    # Iterate k from 3 upwards
    time_alloc = {
        3: 10.0,
        4: 10.0,
        5: 10.0,
        6: 5.0,
        7: 5.0,
        8: 5.0,
        9: 5.0,
        10: 5.0,
        11: 5.0,
        12: 5.0
    }
    
    for k in range(3, 13):
        if time.time() > deadline - 5.0:
            break
        t_limit = time_alloc.get(k, 2.0)
        result = search_k_colors(k, t_limit)
        if result is not None:
            return result.tolist()
            
    # Fallback: Greedy approach (only checking 4-AP constraints)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                idx1 = x - 1 - d
                idx2 = x - 1 - 2 * d
                idx3 = x - 1 - 3 * d
                if idx3 < 0:
                    continue
                c_set = {color, colors[idx1], colors[idx2], colors[idx3]}
                if len(c_set) < 3:
                    valid = False
                    break
            
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
            
    return colors
