"""Rewritten program: randomized local search with incremental violation tracking for 4-AP coloring."""
import time
import random
import sys

def construct():
    """Construct a coloring of positions 1..128 minimizing distinct colors,
    with every nonconstant 4-term AP using >= 3 distinct colors."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    n = 128
    
    # Precompute all nonconstant 4-term arithmetic progressions (a, a+d, a+2d, a+3d) within 1..128
    # Store as 0-indexed tuples (i, i+d, i+2d, i+3d)
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps.append((a - 1, b - 1, c - 1, e - 1))
    
    num_aps = len(aps)
    
    def count_violations(colors):
        """Count the number of 4-APs that use < 3 distinct colors."""
        violations = 0
        for (i, j, k, l) in aps:
            colors_in_ap = {colors[i], colors[j], colors[k], colors[l]}
            if len(colors_in_ap) < 3:
                violations += 1
        return violations
    
    def try_local_search(initial_colors, max_iters=50000, timeout=None):
        """Local search: randomly change a position's color, accept if it reduces violations."""
        colors = list(initial_colors)
        current_viol = count_violations(colors)
        if current_viol == 0:
            return colors
        
        # For each position, precompute which APs it belongs to
        pos_to_aps = [[] for _ in range(n)]
        for idx, (i, j, k, l) in enumerate(aps):
            for pos in (i, j, k, l):
                pos_to_aps[pos].append(idx)
        
        # Track violation count per AP
        ap_viol = [False] * num_aps
        for idx, (i, j, k, l) in enumerate(aps):
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                ap_viol[idx] = True
        total_viol = sum(ap_viol)
        
        best_colors = list(colors)
        best_viol = total_viol
        best_num_colors = len(set(colors))
        
        iters = 0
        while iters < max_iters:
            if timeout is not None and time.time() > timeout:
                break
            iters += 1
            
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all possible colors (0..127) and find the best one
            best_new_color = old_color
            best_delta = 0  # change in violation count
            
            for new_color in range(128):
                if new_color == old_color:
                    continue
                
                # Compute delta in violations from changing position pos
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    i, j, k, l = aps[ap_idx]
                    # Build the set of colors in this AP with the new color
                    old_set = {colors[i], colors[j], colors[k], colors[l]}
                    # Replace pos's color
                    new_set = set(old_set)
                    # Remove old color contribution and add new
                    # Simpler: just compute
                    vals = []
                    for p in (i, j, k, l):
                        if p == pos:
                            vals.append(new_color)
                        else:
                            vals.append(colors[p])
                    new_len = len(set(vals))
                    
                    was_viol = ap_viol[ap_idx]
                    is_viol = new_len < 3
                    
                    if was_viol and not is_viol:
                        delta -= 1
                    elif not was_viol and is_viol:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_delta >= 0 and best_new_color == old_color:
                # Try random perturbation: pick random new color
                new_color = random.randint(0, 127)
                if new_color != old_color:
                    delta = 0
                    for ap_idx in pos_to_aps[pos]:
                        i, j, k, l = aps[ap_idx]
                        vals = []
                        for p in (i, j, k, l):
                            if p == pos:
                                vals.append(new_color)
                            else:
                                vals.append(colors[p])
                        new_len = len(set(vals))
                        was_viol = ap_viol[ap_idx]
                        is_viol = new_len < 3
                        if was_viol and not is_viol:
                            delta -= 1
                        elif not was_viol and is_viol:
                            delta += 1
                    
                    if delta < 0:
                        best_delta = delta
                        best_new_color = new_color
            
            if best_new_color == old_color and best_delta == 0:
                # No improvement; do a random move with small probability (simulated annealing-like)
                if random.random() < 0.1:
                    new_color = random.randint(0, 127)
                    if new_color != old_color:
                        # Compute delta
                        delta = 0
                        for ap_idx in pos_to_aps[pos]:
                            i, j, k, l = aps[ap_idx]
                            vals = []
                            for p in (i, j, k, l):
                                if p == pos:
                                    vals.append(new_color)
                                else:
                                    vals.append(colors[p])
                            new_len = len(set(vals))
                            was_viol = ap_viol[ap_idx]
                            is_viol = new_len < 3
                            if was_viol and not is_viol:
                                delta -= 1
                            elif not was_viol and is_viol:
                                delta += 1
                        best_delta = delta
                        best_new_color = new_color
            
            if best_new_color != old_color:
                # Apply the change
                colors[pos] = best_new_color
                for ap_idx in pos_to_aps[pos]:
                    i, j, k, l = aps[ap_idx]
                    vals = [colors[p] for p in (i, j, k, l)]
                    ap_viol[ap_idx] = len(set(vals)) < 3
                total_viol += best_delta
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = list(colors)
                    best_num_colors = len(set(colors))
                
                if total_viol == 0:
                    # Try to merge colors to reduce distinct count
                    used = sorted(set(colors))
                    if len(used) <= 3:
                        return colors
                    # Try to merge two colors
                    for a in range(len(used)):
                        for b in range(a + 1, len(used)):
                            if used[a] == used[b]:
                                continue
                            # Try merging used[b] into used[a]
                            test_colors = list(colors)
                            for p in range(n):
                                if test_colors[p] == used[b]:
                                    test_colors[p] = used[a]
                            v = count_violations(test_colors)
                            if v == 0:
                                return test_colors
                    
            # Check deadline
            if time.time() > deadline:
                break
        
        return best_colors if best_viol == 0 else best_colors
    
    # Strategy 1: Start with random 3-coloring and local search
    best_result = None
    best_score = float('inf')
    
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring with k colors
        k = 3 if attempt < 15 else 4
        initial = [random.randint(0, k - 1) for _ in range(n)]
        
        result = try_local_search(initial, max_iters=20000, timeout=deadline - 5)
        
        if result is not None:
            v = count_violations(result)
            if v == 0:
                num_colors = len(set(result))
                if num_colors < best_score:
                    best_score = num_colors
                    best_result = list(result)
                if best_score == 2:
                    break  # Can't do better than 2 colors (but 2 colors can't work for 4-AP with >=3 distinct)
                if best_score == 3:
                    # Try a few more to see if 3 is achievable consistently
                    if attempt >= 10:
                        break
    
    # If no valid solution found with 3 or 4 colors, try greedy with more colors
    if best_result is None:
        # Greedy construction with minimal colors
        colors = []
        for x in range(1, n + 1):
            found = False
            for color in range(128):
                valid = True
                for (i, j, k, l) in aps:
                    if l == x - 1:  # AP ending at x
                        # Check if adding this color would make the AP have < 3 colors
                        ap_colors = {colors[i], colors[j], colors[k], color}
                        if len(ap_colors) < 3:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Should not happen with 128 colors
                colors.append(random.randint(0, 127))
        
        v = count_violations(colors)
        if v == 0:
            num_colors = len(set(colors))
            if num_colors < best_score:
                best_score = num_colors
                best_result = list(colors)
    
    # Final: try to reduce colors in the best result
    if best_result is not None:
        used = sorted(set(best_result))
        # Try merging colors pairwise
        changed = True
        while changed and len(used) > 3:
            changed = False
            for a in range(len(used)):
                for b in range(a + 1, len(used)):
                    if used[a] not in used or used[b] not in used:
                        continue
                    test = list(best_result)
                    for p in range(n):
                        if test[p] == used[b]:
                            test[p] = used[a]
                    if count_violations(test) == 0:
                        best_result = test
                        used = sorted(set(test))
                        changed = True
                        break
                if changed:
                    break
    
    # Return the best result
    if best_result is not None:
        return best_result
    
    # Fallback: should not reach here
    return list(range(n))
