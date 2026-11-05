import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    
    # Precompute all 4-term APs
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of AP indices
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def is_valid_assignment(pos, color, colors):
        """Check if assigning color to pos is valid given current partial assignment."""
        for ap_idx in pos_to_aps[pos]:
            a, b, c, d_pos = ap_list[ap_idx]
            vals = [colors[a], colors[b], colors[c], colors[d_pos]]
            
            # Set pos's value
            if pos == a:
                vals[0] = color
            elif pos == b:
                vals[1] = color
            elif pos == c:
                vals[2] = color
            else:
                vals[3] = color
            
            # Check if all 4 assigned
            if all(v != -1 for v in vals):
                if len(set(vals)) < 3:
                    return False
            
            # Check if constraint is already violated (3 assigned, all same)
            assigned = [v for v in vals if v != -1]
            if len(assigned) == 3 and len(set(assigned)) == 1:
                return False
        
        return True
    
    # Greedy: minimize colors by preferring already-used colors
    colors = [-1] * n
    colors_used = set()
    
    for pos in range(n):
        # Prefer already-used colors to minimize total
        candidates = list(colors_used)
        for c in range(128):
            if c not in colors_used:
                candidates.append(c)
        
        assigned = False
        for color in candidates:
            if is_valid_assignment(pos, color, colors):
                colors[pos] = color
                colors_used.add(color)
                assigned = True
                break
        
        if not assigned:
            # Fallback: try all colors
            for color in range(128):
                if is_valid_assignment(pos, color, colors):
                    colors[pos] = color
                    colors_used.add(color)
                    break
            else:
                colors[pos] = 0  # Should not happen
    
    return colors
