import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Try to find a valid coloring with K colors
    # Minimum is 4 (3 is impossible)
    
    def solve_with_k(k):
        """Backtracking with forward checking to find valid k-coloring."""
        colors = [-1] * n
        domains = [set(range(k)) for _ in range(n)]
        
        def check_constraint(ap_idx):
            """Check if AP constraint is satisfied or potentially satisfiable."""
            a, b, c, d_pos = ap_list[ap_idx]
            vals = [colors[a], colors[b], colors[c], colors[d_pos]]
            assigned = [v for v in vals if v != -1]
            
            if len(assigned) == 4:
                return len(set(assigned)) >= 3
            
            # If 3 assigned and all same, impossible
            if len(assigned) == 3 and len(set(assigned)) == 1:
                return False
            
            # If 3 assigned and only 2 distinct, the 4th must be a 3rd color
            if len(assigned) == 3 and len(set(assigned)) == 2:
                # The unassigned position must be assigned a color not in the 2 used
                unassigned_pos = [p for p in (a,b,c,d_pos) if colors[p] == -1][0]
                possible_colors = domains[unassigned_pos] - set(assigned)
                return len(possible_colors) > 0
            
            return True
        
        def is_valid(pos, color):
            """Check if assigning color to pos is valid."""
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                # Temporarily assign
                old_color = colors[pos]
                colors[pos] = color
                
                if not check_constraint(ap_idx):
                    colors[pos] = old_color
                    return False
                
                colors[pos] = old_color
            return True
        
        def forward_check(pos, color):
            """After assigning pos=color, check if any unassigned variable has empty domain."""
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                positions = [a, b, c, d_pos]
                unassigned = [p for p in positions if p != pos and colors[p] == -1]
                
                if len(unassigned) == 1:
                    # Only one unassigned, check if it can be satisfied
                    p = unassigned[0]
                    vals = [colors[x] for x in positions if x != p]
                    # Need len(set(vals + [new_color])) >= 3
                    # So new_color must make the set size >= 3
                    current_distinct = len(set(vals))
                    
                    if current_distinct >= 3:
                        # Any color works
                        continue
                    elif current_distinct == 2:
                        # Need a 3rd color
                        used_colors = set(vals)
                        possible = domains[p] - used_colors
                        if not possible:
                            return False
                    elif current_distinct == 1:
                        # Need 2 more distinct colors, but only 1 position left
                        # Impossible
                        return False
                    else:
                        # current_distinct == 0, need 3 distinct from 1 position
                        # Impossible
                        return False
                
                elif len(unassigned) == 2:
                    # Two unassigned positions
                    # We need the final set to have >= 3 distinct colors
                    # This is always possible if k >= 3, but let's be careful
                    # For now, don't prune
                    pass
                
                elif len(unassigned) == 3:
                    # Three unassigned, always possible with k>=3
                    pass
            
            return True
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # MRV: find unassigned variable with smallest domain
            min_domain_size = float('inf')
            min_pos = -1
            for i in range(n):
                if colors[i] == -1 and len(domains[i]) < min_domain_size:
                    min_domain_size = len(domains[i])
                    min_pos = i
                    if min_domain_size == 1:
                        break
            
            if min_pos == -1:
                # All assigned
                return True
            
            # Try colors in domain of min_pos
            for color in sorted(domains[min_pos]):
                if color not in domains[min_pos]:
                    continue
                    
                colors[min_pos] = color
                domains[min_pos].discard(color)
                
                if is_valid(min_pos, color) and forward_check(min_pos, color):
                    if backtrack():
                        return True
                
                colors[min_pos] = -1
                domains[min_pos].add(color)
            
            return False
        
        return backtrack()
    
    # Try with 4 colors first
    if solve_with_k(4):
        # Return the coloring
        return [c if c != -1 else 0 for c in colors]
    
    # Fallback: greedy
    colors = []
    for x in range(1, n + 1):
        idx = x - 1
        best_color = -1
        for color in range(128):
            valid = True
            for d in range(1, (idx // 3) + 1):
                a = idx - 3 * d
                c1 = colors[a]
                c2 = colors[a + d]
                c3 = colors[a + 2 * d]
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            for d in range(1, (idx // 2) + 1):
                a = idx - 2 * d
                c1 = colors[a]
                c2 = colors[a + d]
                if color == c1 and c1 == c2:
                    valid = False
                    break
            if valid:
                best_color = color
                break
        if best_color == -1:
            best_color = 0
        colors.append(best_color)
    
    return colors
