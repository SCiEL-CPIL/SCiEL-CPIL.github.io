import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    N = 128
    # Precompute all nonconstant 4-term progressions
    # A 4-AP is defined by start a and difference d > 0
    # a, a+d, a+2d, a+3d <= N
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            ap_id = len(aps)
            aps.append(idx)
            for p in idx:
                pos_to_aps[p].append(ap_id)
    
    num_aps = len(aps)
    
    def count_violations(colors):
        """Count 4-APs with < 3 distinct colors."""
        v = 0
        for ap in aps:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                v += 1
        return v

    def get_violation_status(colors):
        """Return a boolean array where True means the AP is a violation."""
        status = np.zeros(num_aps, dtype=bool)
        for i, ap in enumerate(aps):
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                status[i] = True
        return status

    def try_coloring(k, time_limit, seed=42):
        """Try to find a valid coloring with k colors using simulated annealing."""
        rng = random.Random(seed)
        start_time = time.time()
        colors = np.array([rng.randint(0, k) for _ in range(N)])
        
        # Initialize violation status
        v_status = get_violation_status(colors)
        v_count = np.sum(v_status)
        
        best_v = v_count
        best_colors = colors.copy()
        
        # Simulated Annealing parameters
        T = 1.0
        T_min = 0.001
        alpha = 0.9995 # cooling rate
        
        while time.time() - start_time < time_limit:
            # Pick random position
            p = rng.randint(0, N - 1)
            old_c = colors[p]
            
            # Pick new color
            new_c = rng.randint(0, k)
            if new_c == old_c:
                continue
                
            # Calculate change in violations
            delta_v = 0
            affected_aps = pos_to_aps[p]
            
            # We need to check if the change reduces or increases violations
            # For each affected AP, check old status and new status
            for ap_id in affected_aps:
                ap = aps[ap_id]
                # Get indices of the other 3 elements
                others = [x for x in ap if x != p]
                
                # Old colors in AP
                old_colors = set([colors[x] for x in others] + [old_c])
                old_viol = len(old_colors) < 3
                
                # New colors in AP
                new_colors = set([colors[x] for x in others] + [new_c])
                new_viol = len(new_colors) < 3
                
                if not old_viol and new_viol:
                    delta_v += 1
                elif old_viol and not new_viol:
                    delta_v -= 1
                
            # Acceptance criterion
            if delta_v < 0:
                # Always accept improvement
                colors[p] = new_c
                for ap_id in affected_aps:
                    ap = aps[ap_id]
                    others = [x for x in ap if x != p]
                    c = set([colors[x] for x in others] + [new_c])
                    v_status[ap_id] = len(c) < 3
                v_count += delta_v
                if v_count < best_v:
                    best_v = v_count
                    best_colors = colors.copy()
                    if v_count == 0:
                        return best_colors
            else:
                # Accept with probability exp(-delta_v / T)
                if delta_v == 0 or rng.random() < np.exp(-delta_v / T):
                    colors[p] = new_c
                    for ap_id in affected_aps:
                        ap = aps[ap_id]
                        others = [x for x in ap if x != p]
                        c = set([colors[x] for x in others] + [new_c])
                        v_status[ap_id] = len(c) < 3
                    v_count += delta_v
                    if v_count < best_v:
                        best_v = v_count
                        best_colors = colors.copy()
                        if v_count == 0:
                            return best_colors
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_colors

    # Deadline guard
    total_start = time.time()
    total_limit = 85.0 # Leave 5s margin for safety

    best_result = None
    best_violations = float('inf')

    # Try 3 colors first with multiple restarts
    if time.time() - total_start < total_limit:
        for seed in range(10):
            if time.time() - total_start >= total_limit - 5:
                break
            res = try_coloring(3, time_limit=5.0, seed=seed)
            v = count_violations(res)
            if v == 0:
                return res.tolist()
            if v < best_violations:
                best_violations = v
                best_result = res

    # If 3 colors failed, try 4 colors
    if time.time() - total_start < total_limit:
        for seed in range(10):
            if time.time() - total_start >= total_limit - 5:
                break
            res = try_coloring(4, time_limit=5.0, seed=seed)
            v = count_violations(res)
            if v == 0:
                return res.tolist()
            if v < best_violations:
                best_violations = v
                best_result = res

    # If 4 colors failed, try 5 colors
    if time.time() - total_start < total_limit:
        for seed in range(10):
            if time.time() - total_start >= total_limit - 5:
                break
            res = try_coloring(5, time_limit=5.0, seed=seed)
            v = count_violations(res)
            if v == 0:
                return res.tolist()
            if v < best_violations:
                best_violations = v
                best_result = res

    # Fallback: Return the best result found so far
    if best_result is not None:
        return best_result.tolist()

    # Last resort: Greedy construction
    colors = []
    for x in range(1, N + 1):
        found = False
        for color in range(N):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors

# EVOLVE-BLOCK-END
