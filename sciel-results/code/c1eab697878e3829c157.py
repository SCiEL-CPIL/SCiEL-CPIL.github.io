import itertools

N = 16

def construct():
    # Initialize with the 9-color solution
    # Pairs: (0,1), (2,3), ..., (14,15)
    # Internal edges get colors 1..8
    # All other edges get color 0
    
    colors = [0] * (N*(N-1)//2)
    
    # Map edge (i,j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        # Number of edges before row i
        # Edges in rows 0..i-1
        # Row k has edges (k, k+1), ..., (k, N-1). Count = N-1-k
        # Sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
        return i*(N-1) - i*(i-1)//2 + (j - i - 1)
        
    # Assign internal colors
    for p in range(8):
        u = 2*p
        v = 2*p + 1
        idx = edge_idx(u, v)
        colors[idx] = p + 1
        
    used_colors = set(colors)
    
    # Helper to check validity of a specific edge change
    # This is slow if done naively, so we precompute cycles?
    # Or just check all 4-cycles involving the edge.
    
    def is_valid_change(edge_u, edge_v, new_color):
        # Get the current color
        idx = edge_idx(edge_u, edge_v)
        old_color = colors[idx]
        if new_color == old_color:
            return True
            
        # Temporarily set new color
        colors[idx] = new_color
        
        # Find all 4-cycles involving (edge_u, edge_v)
        # A 4-cycle is a set of 4 vertices.
        # The edge (edge_u, edge_v) is one of the edges.
        # The other 2 vertices must be chosen from the remaining N-2 vertices.
        valid = True
        for x, y in itertools.combinations([v for v in range(N) if v != edge_u and v != edge_v], 2):
            # Vertices are edge_u, edge_v, x, y
            # The 3 cycles on these 4 vertices are:
            # 1. edge_u - edge_v - x - y - edge_u
            # 2. edge_u - edge_v - y - x - edge_u
            # 3. edge_u - x - edge_v - y - edge_u
            
            # Cycle 1: (edge_u, edge_v), (edge_v, x), (x, y), (y, edge_u)
            c1 = [
                new_color,
                colors[edge_idx(edge_v, x)],
                colors[edge_idx(x, y)],
                colors[edge_idx(y, edge_u)]
            ]
            if len(set(c1)) == 4:
                valid = False
                break
                
            # Cycle 2: (edge_u, edge_v), (edge_v, y), (y, x), (x, edge_u)
            c2 = [
                new_color,
                colors[edge_idx(edge_v, y)],
                colors[edge_idx(y, x)],
                colors[edge_idx(x, edge_u)]
            ]
            if len(set(c2)) == 4:
                valid = False
                break
                
            # Cycle 3: (edge_u, x), (x, edge_v), (edge_v, y), (y, edge_u)
            # Note: (edge_v, y) is same as (y, edge_v)
            c3 = [
                colors[edge_idx(edge_u, x)],
                colors[edge_idx(x, edge_v)],
                colors[edge_idx(edge_v, y)],
                colors[edge_idx(y, edge_u)]
            ]
            if len(set(c3)) == 4:
                valid = False
                break
                
        # Revert
        colors[idx] = old_color
        
        return valid

    # Greedy improvement
    # Try to assign new colors to edges
    # We iterate multiple times
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                # Try all colors not in used_colors
                # Also try colors in used_colors? No, we want to maximize count.
                # But maybe changing to an existing color allows a new color later?
                # For now, stick to adding new colors.
                for c in range(120):
                    if c in used_colors:
                        continue
                    if is_valid_change(u, v, c):
                        # Accept
                        idx = edge_idx(u, v)
                        colors[idx] = c
                        used_colors.add(c)
                        improved = True
                        # Break to restart? Or continue?
                        # Continue to see if more can be added
                        break # Only add one new color per edge in this pass?
                        # Actually, if we break, we only try one new color.
                        # We should try all new colors to find the best one?
                        # No, we just want to increase the count. Any new color is good.
                        # But maybe a different new color allows more later.
                        # Let's just accept the first valid new color.
    
    return colors
