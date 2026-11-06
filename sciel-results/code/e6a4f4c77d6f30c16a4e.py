"""Colorings with three colors in every four-term progression"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs and index them by position
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
    
    num_aps = len(aps)
    
    # Convert to numpy for fast batch operations
    aps_np = np.array(aps, dtype=np.int32)
    
    def is_violation(colors, i, j, k, l):
        """Check if a specific AP is a violation (fewer than 3 distinct colors)"""
        c0, c1, c2, c3 = colors[i], colors[j], colors[k], colors[l]
        # All same
        if c0 == c1 == c2 == c3:
            return True
        # Exactly 2 distinct
        if c0 == c1:
            if c2 == c3:
                return True
            if c2 == c0 or c3 == c0:
                return True
        elif c0 == c2:
            if c1 == c3:
                return True
            if c1 == c0 or c3 == c0:
                return True
        elif c0 == c3:
            if c1 == c2:
                return True
            if c1 == c0 or c2 == c0:
                return True
        elif c1 == c2:
            if c0 == c3:
                return True
            if c0 == c1 or c3 == c1:
                return True
        elif c1 == c3:
            if c0 == c2:
                return True
            if c0 == c1 or c2 == c1:
                return True
        elif c2 == c3:
            if c0 == c1:
                return True
            if c0 == c2 or c1 == c2:
                return True
        return False
    
    def compute_violations(colors):
        """Return set of AP indices that are violations"""
        viols = set()
        for idx, (i, j, k, l) in enumerate(aps):
            if is_violation(colors, i, j, k, l):
                viols.add(idx)
        return viols
    
    best_colors = None
    best_distinct = 128
    
    # Try different numbers of colors, starting small
    for max_k in range(3, 15):
        if time.time() > deadline - 10:
            break
        
        num_trials = 200 if max_k <= 4 else 100
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initial coloring
            colors = [random.randint(0, max_k - 1) for _ in range(N)]
            viol_set = compute_violations(colors)
            cur_viol_count = len(viol_set)
            
            if cur_viol_count == 0:
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors[:]
                if best_distinct <= max_k:
                    break
                continue
            
            # Local search with incremental updates
            no_improve = 0
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > deadline - 3:
                    break
                if cur_viol_count == 0:
                    distinct = len(set(colors))
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = colors[:]
                    break
                
                # Pick a random position
                pos = random.randint(0, N - 1)
                old_color = colors[pos]
                
                # Try each possible new color and compute delta
                best_delta = 0
                best_new_color = -1
                for new_color in range(max_k):
                    if new_color == old_color:
                        continue
                    
                    # Compute delta in violation count
                    delta = 0
                    for ap_idx in ap_indices[pos]:
                        i, j, k, l = aps[ap_idx]
                        was_viol = ap_idx in viol_set
                        # Temporarily set
                        colors[pos] = new_color
                        is_viol = is_violation(colors, i, j, k, l)
                        colors[pos] = old_color
                        
                        if was_viol and not is_viol:
                            delta -= 1
                        elif not was_viol and is_viol:
                            delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                    elif delta == best_delta and best_new_color == -1:
                        best_new_color = new_color
                
                if best_new_color == -1:
                    continue
                
                # Accept if improves or with some probability if neutral
                if best_delta < 0 or (best_delta == 0 and random.random() < 0.2):
                    # Apply the change
                    colors[pos] = best_new_color
                    # Update viol_set
                    for ap_idx in ap_indices[pos]:
                        i, j, k, l = aps[ap_idx]
                        was_viol = ap_idx in viol_set
                        is_viol = is_violation(colors, i, j, k, l)
                        if was_viol and not is_viol:
                            viol_set.discard(ap_idx)
                        elif not was_viol and is_viol:
                            viol_set.add(ap_idx)
                    cur_viol_count = len(viol_set)
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 300:
                    break
            
            if cur_viol_count == 0:
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors[:]
        
        if best_distinct <= max_k:
            break
    
    # Fallback: greedy construction
    if best_colors is None:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                valid = True
                # Check all 4-term APs ending at x
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a >= 1:
                        # AP: a, a+d, a+2d, x
                        existing = [colors[a - 1], colors[a + d - 1], colors[a + 2 * d - 1]]
                        if len({color} | set(existing)) < 3:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(0)  # Should not happen
        return colors
    
    # Remap to 0..distinct-1
    color_map = {}
    result = []
    for c in best_colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        result.append(color_map[c])
    
    return result
# EVOLVE-BLOCK-END
