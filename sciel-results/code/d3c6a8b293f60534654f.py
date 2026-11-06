def construct():
    n = 128
    # Try to find a coloring with minimal colors
    # First try 3 colors with backtracking
    import time
    start_time = time.time()
    
    def check_constraint(colors, pos, color):
        # Check if assigning color to pos (1-indexed) violates any 4-term AP
        # A 4-term AP ending at pos: (pos-3d, pos-2d, pos-d, pos) for d such that pos-3d >= 1
        # Also APs where pos is not the last element
        # Actually, we need to check all 4-term APs that include pos
        # For a 4-term AP (a, a+d, a+2d, a+3d), pos could be at position 1,2,3,4
        for d in range(1, n):
            # pos is the 4th term: a = pos - 3d
            a = pos - 3*d
            if a >= 1:
                terms = [a, a+d, a+2*d, a+3*d]
                cols = [color]
                for t in terms[:-1]:
                    if t < pos:
                        cols.append(colors[t-1])
                    else:
                        # This shouldn't happen if we're filling left to right
                        return False
                if len(set(cols)) < 3:
                    return False
            
            # pos is the 3rd term: a = pos - 2d, 4th term = pos + d
            a = pos - 2*d
            if a >= 1 and pos + d <= n:
                # We can't check future terms yet, so skip or handle differently
                # For greedy left-to-right, we only check APs fully determined by past + current
                pass
            
            # pos is the 2nd term: a = pos - d, 3rd = pos + d, 4th = pos + 2d
            a = pos - d
            if a >= 1 and pos + 2*d <= n:
                pass  # future terms
            
            # pos is the 1st term: future terms
            if pos + 3*d <= n:
                pass  # future terms
        
        return True
    
    # Better: only check APs where all 4 terms are <= pos (i.e., pos is the last term)
    # and APs where pos is one of the first 3 terms but we can't fully check yet
    # For a valid greedy, we need to ensure no AP with all 4 terms <= current position is violated
    
    def check_past_aps(colors, pos, color):
        """Check all 4-term APs where the maximum index is pos"""
        # pos is the 4th term: (pos-3d, pos-2d, pos-d, pos)
        for d in range(1, (pos-1)//3 + 1):
            a = pos - 3*d
            if a < 1:
                break
            c1 = colors[a-1]
            c2 = colors[a+d-1]
            c3 = colors[a+2*d-1]
            c4 = color
            if len({c1,c2,c3,c4}) < 3:
                return False
        
        # pos is the 3rd term: (pos-2d, pos-d, pos, pos+d) - but pos+d > pos, so not all past
        # We can't check these yet. But we need to ensure that when pos+d is filled, the AP is valid.
        # For now, we only enforce constraints on APs fully in the past.
        
        # pos is the 2nd term: (pos-d, pos, pos+d, pos+2d) - future terms
        # pos is the 1st term: (pos, pos+d, pos+2d, pos+3d) - future terms
        
        return True
    
    # This greedy only checks APs where pos is the last term. This may not be sufficient.
    # Let me use a more careful approach: after assigning, check all APs that are now fully assigned.
    
    def is_valid_assignment(colors, pos, color):
        """Check if assigning color to position pos (1-indexed) is valid,
        considering all 4-term APs that are now fully assigned (all indices <= pos)"""
        # For each 4-term AP (a, a+d, a+2d, a+3d) where a+3d == pos (pos is last)
        for d in range(1, (pos-1)//3 + 1):
            a = pos - 3*d
            if a < 1:
                break
            c = [colors[a-1], colors[a+d-1], colors[a+2*d-1], color]
            if len(set(c)) < 3:
                return False
        
        # Also check APs where pos is the 3rd term: (a, a+d, a+2d, a+3d) with a+2d = pos, so a+3d = pos+d > pos
        # These are not fully assigned yet, so we can't check. But we need to ensure they CAN be valid.
        # For simplicity, let's only check fully-assigned APs for now and see if it works.
        
        return True
    
    # Try greedy with 3 colors
    for num_colors in [3, 4, 5, 6]:
        colors = [-1] * n
        success = True
        for pos in range(1, n+1):
            found = False
            for c in range(num_colors):
                if is_valid_assignment(colors, pos, c):
                    colors[pos-1] = c
                    found = True
                    break
            if not found:
                success = False
                break
        if success:
            # Verify the full coloring
            valid = True
            for a in range(1, n+1):
                for d in range(1, n+1):
                    if a + 3*d > n:
                        break
                    c = [colors[a-1], colors[a+d-1], colors[a+2*d-1], colors[a+3*d-1]]
                    if len(set(c)) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return colors
    
    # Fallback: use the seed approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
