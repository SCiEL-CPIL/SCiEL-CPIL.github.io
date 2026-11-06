"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations_fast(colors):
        """Count violations using bitmask for speed."""
        v = 0
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            # Check if fewer than 3 distinct colors
            if c0 == c1 and c0 == c2 and c0 == c3:
                v += 1
            elif c0 == c1 and c0 == c2:
                pass  # 2 distinct if c3 != c0
            elif c0 == c1 and c0 == c3:
                pass
            elif c0 == c2 and c0 == c3:
                pass
            elif c1 == c2 and c1 == c3:
                pass
            elif c0 == c1:
                if c2 == c3:
                    # Check if only 2 distinct
                    if c0 != c2:
                        pass  # 2 distinct, valid
                    else:
                        v += 1  # 1 distinct
                else:
                    pass  # at least 3 distinct
            elif c0 == c2:
                if c1 == c3:
                    if c0 != c1:
                        pass
                    else:
                        v += 1
                else:
                    pass
            elif c0 == c3:
                if c1 == c2:
                    if c0 != c1:
                        pass
                    else:
                        v += 1
                else:
                    pass
            elif c1 == c2:
                if c0 == c3:
                    if c1 != c0:
                        pass
                    else:
                        v += 1
                else:
                    pass
            elif c1 == c3:
                if c0 == c2:
                    if c1 != c0:
                        pass
                    else:
                        v += 1
                else:
                    pass
            elif c2 == c3:
                if c0 == c1:
                    if c2 != c0:
                        pass
                    else:
                        v += 1
                else:
                    pass
            else:
                pass  # all distinct
        return v
    
    def is_ap_valid_fast(colors, ap):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        # Valid if at least 3 distinct colors
        if c0 == c1 == c2 == c3:
            return False
        # Count distinct
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        return distinct >= 3
    
    def greedy_construct():
        """Deterministic greedy: assign smallest valid color at each position."""
        colors = [0] * n
        for x in range(n):
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    # Check if all other positions are assigned
                    all_assigned = True
                    for pos in ap:
                        if pos != x and colors[pos] == 0:
                            # Position not yet assigned, can't check
                            all_assigned = False
                            break
                    if not all_assigned:
                        continue
                    # Check validity with candidate color
                    if not is_ap_valid_fast_with_candidate(colors, ap, x, color):
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
        return colors
    
    def is_ap_valid_fast_with_candidate(colors, ap, x, candidate):
        """Check if AP is valid when position x has the candidate color."""
        c0 = colors[ap[0]] if ap[0] != x else candidate
        c1 = colors[ap[1]] if ap[1] != x else candidate
        c2 = colors[ap[2]] if ap[2] != x else candidate
        c3 = colors[ap[3]] if ap[3] != x else candidate
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        return distinct >= 3
    
    def try_reduce_colors(colors, max_time):
        """Try to reduce colors by remapping."""
        colors = colors[:]
        best_colors = colors[:]
        best_count = len(set(colors))
        
        while time.time() < max_time:
            used_colors = sorted(set(colors))
            if len(used_colors) <= best_count:
                break
            
            improved = False
            for color_to_remove in used_colors:
                if time.time() > max_time:
                    break
                
                positions = [i for i in range(n) if colors[i] == color_to_remove]
                if not positions:
                    continue
                
                for target_color in used_colors:
                    if target_color == color_to_remove:
                        continue
                    
                    # Try remapping
                    old_vals = []
                    for pos in positions:
                        old_vals.append((pos, colors[pos]))
                        colors[pos] = target_color
                    
                    v = count_violations_fast(colors)
                    if v == 0:
                        best_colors = colors[:]
                        best_count = len(set(colors))
                        improved = True
                        break
                    else:
                        for pos, val in old_vals:
                            colors[pos] = val
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_colors, best_count
    
    def local_search_refine(colors, time_limit):
        """Simulated annealing with better candidate selection."""
        colors = colors[:]
        current_violations = count_violations_fast(colors)
        current_colors_used = len(set(colors))
        
        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        temp = 5.0
        cooling = 0.99
        min_temp = 0.1
        
        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Get used colors
            used = list(set(colors))
            # Prioritize colors that appear less frequently (to encourage consolidation)
            color_freq = {}
            for c in colors:
                color_freq[c] = color_freq.get(c, 0) + 1
            # Sort by frequency (ascending) to try rare colors first
            used_sorted = sorted(used, key=lambda c: color_freq[c])
            
            # Build candidate list: all used colors + a few random
            candidates = used_sorted[:]
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)
            
            best_new_color = old_color
            best_score = float('inf')
            
            affected_aps = pos_to_aps[pos]
            
            # Count current violations among affected APs
            curr_invalid = 0
            for ap_idx in affected_aps:
                if not is_ap_valid_fast(colors, aps[ap_idx]):
                    curr_invalid += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_invalid = 0
                for ap_idx in affected_aps:
                    if not is_ap_valid_fast(colors, aps[ap_idx]):
                        new_invalid += 1
                
                delta_v = new_invalid - curr_invalid
                new_used = len(set(colors))
                delta_c = new_used - current_colors_used
                
                # Weighted score: violations dominate, then color count
                score = delta_v * 1000 + delta_c
                
                if score < best_score:
                    best_score = score
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                elif score == best_score and random.random() < 0.2:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
            
            colors[pos] = old_color
            
            # Acceptance
            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                energy = best_delta_v * 1000 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
            
            if accept:
                colors[pos] = best_new_color
                current_violations += best_delta_v
                current_colors_used = len(set(colors))
                
                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]
                
                if current_violations == 0 and current_colors_used <= 3:
                    break
            
            temp *= cooling
        
        return best_colors, best_violations, best_colors_used
    
    # Step 1: Greedy construction
    colors = greedy_construct()
    initial_violations = count_violations_fast(colors)
    initial_colors = len(set(colors))
    
    # Step 2: Try to reduce colors
    if initial_violations == 0:
        colors, reduced_count = try_reduce_colors(colors, time.time() + 30.0)
        initial_violations = count_violations_fast(colors)
        initial_colors = len(set(colors))
    
    # Step 3: Local search refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        colors, final_violations, final_colors = local_search_refine(colors, deadline - 2.0)
    else:
        final_violations = initial_violations
        final_colors = initial_colors
    
    # Ensure valid solution
    if final_violations > 0:
        return list(range(n))
    
    return colors
# EVOLVE-BLOCK-END
