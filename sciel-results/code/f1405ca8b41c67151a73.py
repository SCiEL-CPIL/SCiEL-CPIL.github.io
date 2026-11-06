"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 128
    # Precompute all 4-term APs ending at each position
    # For position i (0-indexed), find all (a, a+d, a+2d, a+3d) where a+3d = i
    ap_endings = [[] for _ in range(n)]
    for i in range(3, n):
        for d in range(1, (i // 3) + 1):
            a = i - 3 * d
            if a >= 0:
                ap_endings[i].append((a, a + d, a + 2 * d, i))
    
    # Also precompute APs where position i is not the last element
    # For forward checking, we need to know constraints on future positions
    # For each position i and color c, check if assigning c creates violations
    # with already-assigned positions in any 4-term AP containing i
    
    def is_valid_assignment(colors, pos, color):
        """Check if assigning 'color' to position 'pos' is valid given current colors."""
        # Check all 4-term APs that include position 'pos'
        # Case 1: pos is the last element (a+3d = pos)
        for (a, a1, a2, a3) in ap_endings[pos]:
            if colors[a] != -1 and colors[a1] != -1 and colors[a2] != -1:
                if len({color, colors[a], colors[a1], colors[a2]}) < 3:
                    return False
        
        # Case 2: pos is not the last element
        # pos = a: AP is (pos, pos+d, pos+2d, pos+3d)
        for d in range(1, (n - 1 - pos) // 3 + 1):
            a1 = pos + d
            a2 = pos + 2 * d
            a3 = pos + 3 * d
            if a3 < n:
                assigned = []
                if colors[a1] != -1:
                    assigned.append(colors[a1])
                if colors[a2] != -1:
                    assigned.append(colors[a2])
                if colors[a3] != -1:
                    assigned.append(colors[a3])
                if len(assigned) >= 2:
                    if len({color} | set(assigned)) < 3:
                        return False
        
        # pos = a+d: AP is (a, pos, a+2d, a+3d)
        for d in range(1, pos + 1):
            a = pos - d
            a2 = pos + d
            a3 = pos + 2 * d
            if a3 < n:
                assigned = []
                if colors[a] != -1:
                    assigned.append(colors[a])
                if colors[a2] != -1:
                    assigned.append(colors[a2])
                if colors[a3] != -1:
                    assigned.append(colors[a3])
                if len(assigned) >= 2:
                    if len({color} | set(assigned)) < 3:
                        return False
        
        # pos = a+2d: AP is (a, a+d, pos, a+3d)
        for d in range(1, (pos // 2) + 1):
            a = pos - 2 * d
            a1 = pos - d
            a3 = pos + d
            if a3 < n:
                assigned = []
                if colors[a] != -1:
                    assigned.append(colors[a])
                if colors[a1] != -1:
                    assigned.append(colors[a1])
                if colors[a3] != -1:
                    assigned.append(colors[a3])
                if len(assigned) >= 2:
                    if len({color} | set(assigned)) < 3:
                        return False
        
        return True
    
    # Try to solve with increasing number of colors
    for num_colors in range(3, 10):
        if time.time() > deadline:
            break
        colors = [-1] * n
        colors[0] = 0
        colors[1] = 0 if num_colors > 1 else 0
        colors[2] = 0 if num_colors > 2 else (1 if num_colors > 1 else 0)
        
        # Use backtracking with forward checking
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            # Try colors in order, preferring already-used colors
            used_colors = set(colors[:pos])
            color_order = list(used_colors) + [c for c in range(num_colors) if c not in used_colors]
            
            for color in color_order:
                if is_valid_assignment(colors, pos, color):
                    colors[pos] = color
                    if backtrack(pos + 1):
                        return True
                    colors[pos] = -1
            
            return False
        
        if backtrack(3):
            # Extract actual colors used
            result = [colors[i] for i in range(n)]
            return result
    
    # Fallback: greedy construction
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
# EVOLVE-BLOCK-END
