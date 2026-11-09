"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps)
    
    # For each position, which AP indices does it appear in?
    pos_ap_list = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_ap_list[a].append(idx)
        pos_ap_list[b].append(idx)
        pos_ap_list[c].append(idx)
        pos_ap_list[e].append(idx)
    
    # For each AP, store the other 3 positions for each position in it
    # ap_other[ap_idx][pos_in_ap] = list of other 3 positions
    ap_other = {}
    for idx, (a, b, c, e) in enumerate(aps):
        ap_other[idx] = {a: (b, c, e), b: (a, c, e), c: (a, b, e), e: (a, b, c)}
    
    # Try 3 colors first
    for num_colors in [3, 4, 5]:
        if time.time() > deadline - 2.0:
            break
        
        colors = [-1] * n
        # domains[pos] = bitmask of valid colors (bit i set means color i is valid)
        all_mask = (1 << num_colors) - 1
        domains = [all_mask] * n
        
        def count_bits(x):
            return bin(x).count('1')
        
        def check_ap_constraint(ap_idx, pos, color_bit):
            """After assigning color to pos, check if AP constraint is still satisfiable."""
            a, b, c, e = aps[ap_idx]
            others = ap_other[ap_idx][pos]
            # Get colors of the other 3 positions
            oc = []
            for o in others:
                if colors[o] >= 0:
                    oc.append(colors[o])
                else:
                    oc.append(-1)
            
            # The 4 colors are: color (for pos) + the 3 others
            # We need at least 3 distinct colors among all 4
            # If all 4 are assigned, check directly
            if all(o >= 0 for o in oc):
                all_c = [color] + oc
                return len(set(all_c)) >= 3
            
            # Some are unassigned - check if it's still possible
            # The assigned colors among the 4:
            assigned_colors = [color]
            for o in oc:
                if o >= 0:
                    assigned_colors.append(o)
            
            num_assigned = len(assigned_colors)
            distinct_assigned = len(set(assigned_colors))
            
            if num_assigned == 4:
                return distinct_assigned >= 3
            
            # We need at least 3 distinct total
            # If distinct_assigned >= 3, already satisfied
            if distinct_assigned >= 3:
                return True
            
            # If distinct_assigned == 2, we need at least one unassigned to take a 3rd color
            # If distinct_assigned == 1, we need at least two unassigned to take new colors
            num_unassigned = 4 - num_assigned
            if distinct_assigned == 2:
                # Need at least 1 unassigned position to have a color different from the 2 assigned
                # Check if any unassigned position's domain has a color not in assigned_colors
                assigned_set = set(assigned_colors)
                for o in others:
                    if colors[o] < 0:
                        # Check if domain of o has any color outside assigned_set
                        outside = domains[o] & ~sum(1 << c for c in assigned_set)
                        if outside:
                            return True
                return False
            elif distinct_assigned == 1:
                # Need at least 2 more distinct colors from unassigned
                # This is complex; conservatively: need at least 2 unassigned with domains
                # containing colors outside the single assigned color
                assigned_set = set(assigned_colors)
                outside_mask = ~sum(1 << c for c in assigned_set)
                outside_mask &= all_mask
                count_with_outside = 0
                for o in others:
                    if colors[o] < 0:
                        if domains[o] & outside_mask:
                            count_with_outside += 1
                return count_with_outside >= 2
            else:
                # distinct_assigned == 0 shouldn't happen since pos is assigned
                return True
        
        def propagate(pos, color):
            """Assign color to pos, propagate constraints. Returns (success, trail)."""
            trail = []
            domains[pos] = 1 << color
            colors[pos] = color
            trail.append((pos, 1 << color, color))
            
            for ap_idx in pos_ap_list[pos]:
                if not check_ap_constraint(ap_idx, pos, 1 << color):
                    # This AP is violated - need to backtrack
                    return False, trail
                
                # Update domains of other positions in this AP
                a, b, c, e = aps[ap_idx]
                others = ap_other[ap_idx][pos]
                for o in others:
                    if colors[o] < 0:
                        # Check if this AP can still be satisfied with o's current domain
                        # For each color in o's domain, check if it would work
                        new_domain = 0
                        for col in range(num_colors):
                            bit = 1 << col
                            if not (domains[o] & bit):
                                continue
                            # Temporarily check: if o takes col, is AP still satisfiable?
                            # The 4 positions: pos (color), and the other 3
                            others2 = [x for x in (a, b, c, e) if x != pos and x != o]
                            # Check if with pos=color and o=col, the remaining 2 can make >=3 distinct
                            assigned = [color, col]
                            distinct = len(set(assigned))
                            # The other 2 positions
                            possible = True
                            if distinct >= 3:
                                possible = True
                            elif distinct == 2:
                                # Need at least one of the other 2 to have a 3rd color
                                found = False
                                for x in others2:
                                    if colors[x] < 0:
                                        outside = domains[x] & ~sum(1 << c for c in assigned)
                                        if outside:
                                            found = True
                                            break
                                    elif colors[x] not in assigned:
                                        found = True
                                        break
                                possible = found
                            else:  # distinct == 1
                                # Need both other 2 to have colors outside
                                count = 0
                                for x in others2:
                                    if colors[x] < 0:
                                        outside = domains[x] & ~sum(1 << c for c in assigned)
                                        if outside:
                                            count += 1
                                    elif colors[x] not in assigned:
                                        count += 1
                                possible = count >= 2
                            
                            if possible:
                                new_domain |= bit
                        
                        if new_domain == 0:
                            return False, trail
                        if new_domain != domains[o]:
                            trail.append((o, domains[o], -1))
                            domains[o] = new_domain
            
            return True, trail
        
        def backtrack(depth=0):
            if time.time() > deadline:
                return False
            
            # Find unassigned position with smallest domain (MRV)
            best_pos = -1
            best_count = num_colors + 1
            for p in range(n):
                if colors[p] < 0:
                    cnt = count_bits(domains[p])
                    if cnt < best_count:
                        best_count = cnt
                        best_pos = p
                        if cnt == 1:
                            break
            
            if best_pos == -1:
                return True  # All assigned
            
            if best_count == 0:
                return False  # No valid color
            
            # Try colors in order of least constraining (or just 0,1,2)
            for color in range(num_colors):
                bit = 1 << color
                if not (domains[best_pos] & bit):
                    continue
                
                success, trail = propagate(best_pos, color)
                if success:
                    if backtrack(depth + 1):
                        return True
                # Undo trail
                for (p, old_dom, old_col) in reversed(trail):
                    domains[p] = old_dom
                    if old_col >= 0:
                        colors[p] = old_col
                    else:
                        colors[p] = -1
            
            return False
        
        # Initial propagation: try to constrain domains based on empty board
        # Actually, start with all domains = all_mask and just backtrack
        result = backtrack()
        
        if result:
            # Verify
            valid = True
            for a, b, c, e in aps:
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    valid = False
                    break
            if valid:
                num_used = len(set(colors))
                return colors
    
    # Fallback: greedy with many colors
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                prev = [colors[x - 1 - j * d] for j in (1, 2, 3)]
                if len({color} | set(prev)) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(127)
    return colors

# EVOLVE-BLOCK-END
