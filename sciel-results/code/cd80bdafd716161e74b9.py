import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    n = 128
    # Precompute all 4-term APs (a, a+d, a+2d, a+3d) with 0-indexed positions
    # Position i corresponds to value i+1
    aps = []
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                aps.append((a, b, c, e))
    
    # For each position, list of AP indices it appears in
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    # We'll try to find a valid coloring with k colors for k = 3, 4, 5, ...
    # Use backtracking with constraint propagation
    
    def check_ap_valid(ap, colors, k):
        """Check if a 4-term AP has at least 3 distinct colors among assigned positions."""
        a, b, c, e = ap
        assigned_colors = set()
        for pos in (a, b, c, e):
            if colors[pos] is not None:
                assigned_colors.add(colors[pos])
        if len(assigned_colors) >= 3:
            return True
        # If only 1 or 2 distinct colors are assigned, check if it's still possible
        # to reach 3 with remaining unassigned positions
        unassigned = sum(1 for pos in (a, b, c, e) if colors[pos] is None)
        if len(assigned_colors) + unassigned < 3:
            return False
        # If we have 2 distinct colors and 1 unassigned, that unassigned must be a 3rd color
        # If we have 2 distinct colors and 2+ unassigned, still possible
        # If we have 1 distinct color and 1 unassigned, impossible (need 2 more distinct)
        if len(assigned_colors) == 2 and unassigned == 1:
            return True  # The unassigned can be a 3rd color
        if len(assigned_colors) == 1 and unassigned == 1:
            return False  # Can only add 1 more distinct color, total 2
        if len(assigned_colors) == 1 and unassigned == 2:
            return True  # Can add 2 more distinct colors
        if len(assigned_colors) == 0:
            return True
        return True
    
    def backtrack(pos, colors, k, time_limit):
        """Backtracking search for valid k-coloring."""
        if time.time() > time_limit:
            return None
        
        # Find the next unassigned position
        if pos >= n:
            return colors[:]
        
        # Try each color for this position
        for color in range(k):
            colors[pos] = color
            
            # Check all APs involving this position
            valid = True
            for ap_idx in pos_aps[pos]:
                if not check_ap_valid(aps[ap_idx], colors, k):
                    valid = False
                    break
            
            if valid:
                result = backtrack(pos + 1, colors, k, time_limit)
                if result is not None:
                    return result
            
            colors[pos] = None
        
        return None
    
    # Try increasing numbers of colors
    for k in range(3, 20):
        if time.time() > deadline:
            break
        colors = [None] * n
        result = backtrack(0, colors, k, deadline)
        if result is not None:
            # Convert to 0-indexed labels
            return result
    
    # If we couldn't find a small-k solution, fall back to a valid large solution
    # Use the greedy approach as fallback
    colors = []
    for x in range(1, n + 1):
        for color in range(n):
            valid = True
            # Check all 4-term APs ending at position x (1-indexed)
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                b = x - 2 * d
                c = x - d
                if a >= 1 and b >= 1 and c >= 1:
                    ap_colors = {colors[a - 1], colors[b - 1], colors[c - 1], color}
                    if len(ap_colors) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
    
    return colors
