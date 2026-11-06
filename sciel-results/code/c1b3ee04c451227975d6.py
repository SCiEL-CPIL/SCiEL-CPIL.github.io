"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    positions = range(n)
    
    # Precompute all non-constant 4-term APs: (a, a+d, a+2d, a+3d) within [0, n)
    ap_list = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e < n:
                ap_list.append((a, b, c, e))
    
    # For each position, precompute which APs it participates in and its role
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[e].append((idx, 3))
    
    # Initial coloring using greedy algorithm (known valid)
    colors = [0] * n
    # Use the seed greedy to get a valid starting point
    colors_list = []
    for x in range(1, n+1):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                j1, j2, j3 = 1, 2, 3
                idx1 = x - 1 - j1*d
                idx2 = x - 1 - j2*d
                idx3 = x - 1 - j3*d
                if idx3 < 0:
                    continue
                cset = {color, colors_list[idx1], colors_list[idx2], colors_list[idx3]}
                if len(cset) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Also check 3-term subprogressions don't force issues
            valid2 = True
            for d in range(1, (x-1)//2 + 1):
                idx1 = x - 1 - d
                idx2 = x - 1 - 2*d
                if idx2 < 0:
                    continue
                if colors_list[idx1] == color and colors_list[idx2] == color:
                    valid2 = False
                    break
            if valid and valid2:
                colors_list.append(color)
                found = True
                break
        if not found:
            colors_list.append(0)
    
    colors = colors_list[:]
    
    # Count distinct colors
    def count_colors(c):
        return len(set(c))
    
    # Check violations
    def count_violations(c):
        v = 0
        for (a, b, cc, e) in ap_list:
            vals = {c[a], c[b], c[cc], c[e]}
            if len(vals) < 3:
                v += 1
        return v
    
    current_violations = count_violations(colors)
    current_colors = count_colors(colors)
    best_colors = colors[:]
    best_color_count = current_colors
    
    # Simulated annealing
    if time.time() < deadline:
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        step = 0
        
        while time.time() < deadline and T > T_min:
            # Pick a random position to recolor
            pos = random.randint(0, n-1)
            old_color = colors[pos]
            
            # Try a new color (biased toward existing colors to reduce count)
            existing = list(set(colors))
            if random.random() < 0.8:
                new_color = random.choice(existing)
            else:
                new_color = random.randint(0, 127)
            
            if new_color == old_color:
                T *= cooling
                step += 1
                continue
            
            # Compute delta violations
            delta_v = 0
            for (ap_idx, role) in pos_aps[pos]:
                a, b, cc, e = ap_list[ap_idx]
                old_vals = [colors[a], colors[b], colors[cc], colors[e]]
                old_set = set(old_vals)
                old_bad = 1 if len(old_set) < 3 else 0
                
                # Temporarily set new color
                pos_in_ap = [a, b, cc, e]
                new_vals = old_vals[:]
                new_vals[role] = new_color
                new_set = set(new_vals)
                new_bad = 1 if len(new_set) < 3 else 0
                
                delta_v += new_bad - old_bad
            
            # Compute delta color count
            old_count = current_colors
            # Check if old_color is still used elsewhere
            old_color_used = False
            for i in range(n):
                if i != pos and colors[i] == old_color:
                    old_color_used = True
                    break
            new_count = current_colors
            if not old_color_used and colors.count(new_color) == 0:
                new_count = current_colors + 1
            elif old_color_used:
                new_count = current_colors
            else:
                # old_color no longer used, new_color already used
                new_count = current_colors - 1
            
            delta_c = new_count - old_count
            
            # Energy: prioritize no violations, then minimize colors
            K = 1000
            delta_E = K * delta_v + delta_c
            
            if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
                colors[pos] = new_color
                current_violations += delta_v
                current_colors = new_count
                
                if current_violations == 0 and current_colors < best_color_count:
                    best_color_count = current_colors
                    best_colors = colors[:]
            
            T *= cooling
            step += 1
    
    # Return best valid coloring found
    if current_violations == 0:
        return colors
    else:
        return best_colors
# EVOLVE-BLOCK-END
