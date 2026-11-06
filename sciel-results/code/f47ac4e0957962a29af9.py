"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    # AP: (i, i+d, i+2d, i+3d) where d >= 1, all indices < n
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def get_ap_violation(colors, ap):
        """Returns True if the AP violates the condition (uses < 3 colors)."""
        # Inline set creation for speed
        c1 = colors[ap[0]]
        c2 = colors[ap[1]]
        c3 = colors[ap[2]]
        c4 = colors[ap[3]]
        
        # Check if at least 3 distinct colors
        # If all 4 are same -> 1 color
        # If 3 same, 1 diff -> 2 colors
        # If 2 same, 2 same -> 2 colors
        # If 2 same, 1, 1 -> 3 colors (Valid)
        # If 1, 1, 1, 1 -> 4 colors (Valid)
        
        # Efficient check:
        if c1 == c2 == c3 == c4:
            return True
        if c1 == c2 and c3 == c4 and c1 != c3:
            return True
        if c1 == c3 and c2 == c4 and c1 != c2:
            return True
        if c1 == c4 and c2 == c3 and c1 != c2:
            return True
        # Cases with exactly 2 distinct colors:
        # 3 of one, 1 of another
        if (c1 == c2 == c3 and c4 != c1) or (c1 == c2 == c4 and c3 != c1) or (c1 == c3 == c4 and c2 != c1) or (c2 == c3 == c4 and c1 != c2):
            return True
        return False

    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            if get_ap_violation(colors, ap):
                v += 1
        return v

    def greedy_construct():
        """Deterministic greedy: assign smallest valid color at each position."""
        colors = [None] * n
        for x in range(n):
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
                    break
            if colors[x] is None:
                colors[x] = x
        return colors

    def local_search_refine(colors, time_limit):
        """Simulated annealing to fix violations and minimize color count."""
        colors = colors[:]
        
        # Initialize state
        current_violations = count_violations(colors)
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1
        current_colors_used = sum(1 for c in color_counts if c > 0)
        
        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        temp = 5.0
        cooling = 0.999
        min_temp = 0.1
        
        # Precompute used colors list for candidate selection
        used_colors_list = [i for i in range(128) if color_counts[i] > 0]
        
        while time.time() < time_limit and temp > min_temp:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Determine candidate colors: current used colors + a few new ones
            # To minimize colors, we prefer used colors.
            candidates = used_colors_list[:]
            # Add some random new colors to explore if stuck
            for _ in range(5):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)
            
            random.shuffle(candidates)
            
            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0
            best_score = 0
            
            affected_aps = pos_to_aps[pos]
            
            # Count current violations among affected APs
            curr_invalid = 0
            for ap_idx in affected_aps:
                if get_ap_violation(colors, aps[ap_idx]):
                    curr_invalid += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                # Apply change
                colors[pos] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                
                new_invalid = 0
                for ap_idx in affected_aps:
                    if get_ap_violation(colors, aps[ap_idx]):
                        new_invalid += 1
                
                delta_v = new_invalid - curr_invalid
                delta_c = 0
                if color_counts[old_color] == 0:
                    delta_c -= 1 # Removed a color
                if color_counts[new_color] == 1:
                    delta_c += 1 # Added a color
                # Note: if old_color == new_color, delta_c is 0, but we skipped that.
                # If old_color count goes to 0, we lose a color. If new_color count becomes 1, we gain a color.
                
                # Score: prioritize fixing violations, then reducing colors
                # We want to minimize violations, then minimize colors.
                score = delta_v * 1000 + delta_c
                best_score_candidate = best_delta_v * 1000 + best_delta_c
                
                if score < best_score_candidate:
                    best_score_candidate = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                    best_score = score
                elif score == best_score_candidate and random.random() < 0.2:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_score = score
            
            # Revert change to old state before deciding
            colors[pos] = old_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1 # Wait, new_color here is the last one iterated? No, we need to revert to OLD.
            
            # Correction: The loop above changed colors[pos] and counts.
            # We need to revert to OLD_COLOR state.
            # The last iteration left colors[pos] = last_candidate.
            # We must revert to old_color.
            # Let's just re-apply old_color and fix counts.
            # Actually, it's safer to calculate deltas without modifying global state, 
            # or modify and revert carefully.
            
            # Let's do a clean revert:
            # Current state in memory: colors[pos] is whatever the last candidate was.
            # Counts reflect that.
            # We want to go back to old_color.
            last_tried_color = colors[pos]
            color_counts[last_tried_color] -= 1
            color_counts[old_color] += 1
            colors[pos] = old_color
            
            # Now decide acceptance based on best_new_color
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
            
            if accept and best_new_color != old_color:
                # Apply the best move
                colors[pos] = best_new_color
                color_counts[old_color] -= 1
                color_counts[best_new_color] += 1
                
                current_violations += best_delta_v
                current_colors_used += best_delta_c
                
                # Update used_colors_list if needed (lazy update or simple check)
                if color_counts[old_color] == 0:
                    # Remove from list if present
                    if old_color in used_colors_list:
                        used_colors_list.remove(old_color)
                if color_counts[best_new_color] == 1:
                    # Add to list if not present
                    if best_new_color not in used_colors_list:
                        used_colors_list.append(best_new_color)
                
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
    initial_violations = count_violations(colors)
    initial_colors = len(set(colors))
    
    # Step 2: Local search refinement
    # We run local search to try to reduce colors and fix any violations
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        colors, final_violations, final_colors = local_search_refine(colors, deadline - 1.0)
    else:
        final_violations = initial_violations
        final_colors = initial_colors
    
    # Ensure we return a valid solution
    # If the local search didn't find a valid solution, try a few random restarts if time permits
    if final_violations > 0 and time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            random_colors = [random.randint(0, 127) for _ in range(n)]
            rc, rv, rc_used = local_search_refine(random_colors, deadline - 1.0)
            if rv < final_violations or (rv == final_violations and rc_used < final_colors):
                colors = rc
                final_violations = rv
                final_colors = rc_used
            if final_violations == 0:
                break

    if final_violations > 0:
        # Fallback: identity mapping is always valid (uses 128 colors)
        return list(range(n))
    
    return colors
# EVOLVE-BLOCK-END
