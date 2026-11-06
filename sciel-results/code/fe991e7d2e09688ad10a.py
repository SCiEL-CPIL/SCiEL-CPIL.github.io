# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline for 90s limit with safety margin
    
    # Precompute all 4-term arithmetic progressions (non-constant)
    # A 4-term AP is (a, a+d, a+2d, a+3d) with d >= 1
    # Positions are 0-indexed: 0 to 127
    # a + 3d <= 127 => a <= 127 - 3d
    aps = []
    for d in range(1, 43):  # d can be at most 42 (since 3*42=126, a can be 0 to 1)
        for a in range(0, 128 - 3*d):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    n_aps = len(aps)
    
    # Count how many APs each position participates in
    pos_ap_count = [0] * 128
    for ap in aps:
        for p in ap:
            pos_ap_count[p] += 1
    
    # Order positions by constraint tightness (more constraints first)
    # This helps with backtracking
    positions_ordered = sorted(range(128), key=lambda p: -pos_ap_count[p])
    
    # For each position, list which APs it's in and its index in that AP
    pos_aps = [[] for _ in range(128)]
    for idx, ap in enumerate(aps):
        for i, p in enumerate(ap):
            pos_aps[p].append((idx, i))
    
    best_solution = None
    best_num_colors = 128
    
    # Try with increasing number of colors
    for num_colors in range(2, 128):
        if time.time() > deadline:
            break
            
        # If we already found a solution with fewer colors, we're done
        # But we want the minimum, so we stop at the first valid num_colors
        
        colors = [-1] * 128
        found = _solve_backtracking(
            positions_ordered, pos_aps, aps, num_colors, colors, 
            deadline, start_time
        )
        
        if found:
            # Count actual distinct colors used
            distinct = set(c for c in colors if c >= 0)
            actual_colors = len(distinct)
            if best_solution is None or actual_colors < best_num_colors:
                best_num_colors = actual_colors
                best_solution = colors[:]
                # Since we're iterating num_colors in increasing order,
                # the first solution found should use exactly num_colors
                # But let's continue to see if we can do better with same num_colors
                # Actually, if num_colors is the bound, we use at most num_colors
                # Let's return immediately since we're going in order
                break
    
    if best_solution is None:
        # Fallback: use the greedy approach from the original
        return _greedy_construct()
    
    return best_solution

def _greedy_construct():
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

def _solve_backtracking(positions_ordered, pos_aps, aps, num_colors, colors, deadline, start_time):
    """Backtracking solver with constraint propagation."""
    # For each AP, track the colors assigned to its 4 positions
    # An AP is satisfied if it uses at least 3 distinct colors
    
    # We'll use a simpler approach: assign colors one by one in the order
    # and check constraints as we go
    
    assigned = [False] * 128
    
    def is_valid_assignment(pos, color):
        """Check if assigning 'color' to 'pos' violates any constraint with already assigned positions."""
        for ap_idx, i in pos_aps[pos]:
            ap = aps[ap_idx]
            # Check if this AP now has a violation
            # Collect colors of all assigned positions in this AP
            colors_in_ap = []
            valid = True
            for j, p in enumerate(ap):
                if p == pos:
                    c = color
                elif assigned[p]:
                    c = colors[p]
                else:
                    continue
                colors_in_ap.append(c)
            
            # If we have at least 2 colors assigned in this AP, check if they're all the same
            # Actually, we need to ensure that the AP can still use at least 3 distinct colors
            # If all assigned positions in this AP have the same color, and there are 
            # fewer than 3 unassigned positions, we might have a problem
            
            # For now, just check: if all assigned positions in this AP have the same color
            # and the number of unassigned positions is less than 3, then it's invalid
            if colors_in_ap:
                if len(set(colors_in_ap)) == 1:
                    # All assigned have same color
                    unassigned_count = 4 - len(colors_in_ap)
                    if unassigned_count < 3:
                        # Can't get 3 distinct colors
                        return False
        
        return True
    
    def backtrack(idx):
        nonlocal best_solution, best_num_colors
        
        if time.time() > deadline:
            return False
        
        if idx == len(positions_ordered):
            # Check all APs are satisfied
            for ap in aps:
                c_set = set(colors[p] for p in ap)
                if len(c_set) < 3:
                    return False
            return True
        
        pos = positions_ordered[idx]
        assigned[pos] = True
        
        # Try colors in order
        for color in range(num_colors):
            if is_valid_assignment(pos, color):
                colors[pos] = color
                if backtrack(idx + 1):
                    return True
                colors[pos] = -1
        
        assigned[pos] = False
        return False
    
    return backtrack(0)
# EVOLVE-BLOCK-END
