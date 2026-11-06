"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def edge_index(i, j):
        """Map (i,j) with i<j to a linear index in lexicographic order."""
        # Number of edges before row i: i*(i-1)/2
        # In row i, edge (i, j) is at position j-i-1
        return i * (i - 1) // 2 + (j - i - 1)
    
    def has_c6_in_color(colors, n, color):
        """Check if any color class contains a 6-cycle."""
        # Build adjacency for this color
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if colors[edge_index(i, j)] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Check for 6-cycles: v0-v1-v2-v3-v4-v5-v0
        # Use backtracking with symmetry breaking
        for v0 in range(n):
            for v1 in range(v0 + 1, n):
                if not (adj[v0] & (1 << v1)):
                    continue
                for v2 in range(v1 + 1, n):
                    if not (adj[v1] & (1 << v2)):
                        continue
                    for v3 in range(v2 + 1, n):
                        if not (adj[v2] & (1 << v3)):
                            continue
                        for v4 in range(v3 + 1, n):
                            if not (adj[v3] & (1 << v4)):
                                continue
                            for v5 in range(v4 + 1, n):
                                if not (adj[v4] & (1 << v5)):
                                    continue
                                if (adj[v5] & (1 << v0)) and (adj[v0] & (1 << v5)):
                                    return True
        return False
    
    def has_any_c6(colors, n):
        """Check if any color class contains a 6-cycle."""
        for c in range(4):
            if has_c6_in_color(colors, n, c):
                return True
        return False
    
    def backtrack(n, colors, edge_pos):
        """Backtracking search for a valid 4-coloring of K_n."""
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        if edge_pos == n * (n - 1) // 2:
            # Check for C6 in all colors
            if not has_any_c6(colors, n):
                return colors[:]
            return None
        
        i = 0
        j = 0
        # Find which edge this position corresponds to
        # Position p corresponds to some (i,j)
        p = edge_pos
        # Find i such that i*(i-1)/2 <= p < (i+1)*i/2
        i = 0
        while i * (i + 1) // 2 <= p:
            i += 1
        i = i - 1
        j = p - i * (i - 1) // 2 + i + 1
        
        # Try each color with symmetry breaking
        # For the first few edges, restrict colors to break symmetry
        if i == 0 and j == 1:
            # Fix color 0 for edge (0,1) by symmetry
            colors[edge_pos] = 0
            result = backtrack(n, colors, edge_pos + 1)
            if result is not None:
                return result
        elif i == 0 and j == 2:
            # Edge (0,2): try colors 0,1
            for c in range(2):
                colors[edge_pos] = c
                result = backtrack(n, colors, edge_pos + 1)
                if result is not None:
                    return result
        elif i == 0 and j == 3:
            # Edge (0,3): try colors 0,1,2
            for c in range(3):
                colors[edge_pos] = c
                result = backtrack(n, colors, edge_pos + 1)
                if result is not None:
                    return result
        else:
            for c in range(4):
                colors[edge_pos] = c
                # Quick check: don't allow immediate C6
                # Check if adding this edge creates a C6
                # For efficiency, only check when we've placed enough edges
                result = backtrack(n, colors, edge_pos + 1)
                if result is not None:
                    return result
        
        colors[edge_pos] = -1
        return None
    
    # Try increasing n from 9 up to 17
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for n in range(10, 18):
        if time.time() > deadline:
            break
        
        # Number of edges
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        
        result = backtrack(n, colors, 0)
        if result is not None:
            best_n = n
            best_colors = result
            # Continue trying for larger n
        else:
            # No solution found for this n, try next
            continue
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
