import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3

    def local_search(k, time_limit):
        """
        Randomized local search to find a valid coloring with k colors.
        Returns the coloring if found, else None.
        """
        # Initialize random
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Simulated Annealing parameters
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() < time_limit:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all other colors
            best_new_color = old_color
            best_delta = 0
            
            # Calculate current violations for APs containing pos
            affected_aps = pos_to_aps[pos]
            
            # Count how many affected APs are currently invalid
            currently_invalid_count = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    currently_invalid_count += 1
            
            # Try each new color
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Recount violations for affected APs
                new_invalid_count = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid_count += 1
                
                delta = new_invalid_count - currently_invalid_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                elif delta == best_delta and random.random() < 0.5:
                    best_new_color = new_color
            
            # Revert to old color for now, then apply best
            colors[pos] = old_color
            
            # Decide whether to accept
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = True
            else:
                # Accept worse with probability exp(-delta / temp)
                if temp > 0:
                    import math
                    prob = math.exp(-best_delta / temp)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                colors[pos] = best_new_color
                current_violations += best_delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    return colors
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors if best_violations == 0 else None

    def greedy_construct():
        """
        Deterministic greedy: assign smallest valid color at each position.
        This uses the structural idea from inspiration 1.
        """
        colors = [None] * n
        for x in range(n):
            found = False
            for color in range(128):
                valid = True
                # Check all APs that include position x
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    # Build the set of colors in this AP, replacing x's color with candidate
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    found = True
                    break
            if not found:
                # Fallback: should not happen with 128 colors
                colors[x] = x
        return colors

    # Try the greedy construction first
    greedy_result = greedy_construct()
    greedy_violations = count_violations(greedy_result)
    greedy_colors_used = len(set(greedy_result))
    
    best_overall = greedy_result
    best_overall_colors = greedy_colors_used
    
    if greedy_violations == 0:
        # We have a valid solution, try to reduce colors
        # Use local search to try to reduce the number of colors
        for k in range(greedy_colors_used, 2, -1):
            if time.time() > deadline - 5.0:
                break
            
            # Try to find a valid coloring with k colors
            time_slice = 5.0
            result = local_search(k, time.time() + time_slice)
            if result is not None:
                best_overall = result
                best_overall_colors = k
                break
    
    # If greedy didn't work, or we couldn't reduce, try local search with increasing k
    if best_overall_colors > 3:
        k_values = [3, 4, 5, 6, 8, 10, 16, 32, 64, 128]
        
        for k in k_values:
            if time.time() > deadline - 5.0:
                break
            
            # Give each k a time slice, but prioritize smaller k
            time_slice = 10.0 if k <= 4 else 5.0
            if time.time() + time_slice > deadline - 2.0:
                time_slice = max(1.0, deadline - 2.0 - time.time())
            
            result = local_search(k, time.time() + time_slice)
            if result is not None:
                best_overall = result
                best_overall_colors = k
                break
    
    # Fallback: Identity mapping (always valid, uses 128 colors)
    if best_overall_colors == 128:
        return list(range(n))
    
    return best_overall
