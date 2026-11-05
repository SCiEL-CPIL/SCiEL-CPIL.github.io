import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def has_path_len_5(adj_bitmasks, u, v, n):
        """Check if there's a simple path of length 5 from u to v using bitmasks."""
        # Path: u - x1 - x2 - x3 - x4 - v
        # We need to find x1, x2, x3, x4 such that:
        # (u,x1) in adj, (x1,x2) in adj, (x2,x3) in adj, (x3,x4) in adj, (x4,v) in adj
        # All vertices distinct: u, x1, x2, x3, x4, v all different
        
        if u == v:
            return False
        
        # Neighbors of u (excluding v, since v is endpoint at depth 5)
        # Actually we need x1 != v, x1 != u
        # x1 in adj[u], x1 != v
        # x2 in adj[x1], x2 != u, x2 != v
        # x3 in adj[x2], x3 != u, x3 != x1, x3 != v
        # x4 in adj[x3], x4 != u, x4 != x1, x4 != x2, x4 != v
        # v in adj[x4]
        
        # Use bitmask operations
        # adj_bitmasks[i] is a bitmask of neighbors of i
        
        # Step 1: Find all possible x1
        x1_set = adj_bitmasks[u] & ~(1 << u) & ~(1 << v)
        
        # Step 2: For each x1, find possible x2
        # We can do this iteratively with sets
        # For efficiency with n<=20, use sets
        
        # Convert to sets for easier manipulation
        def mask_to_set(mask):
            return {i for i in range(n) if (mask >> i) & 1}
        
        x1_candidates = mask_to_set(x1_set)
        
        for x1 in x1_candidates:
            # x2 in adj[x1], x2 != u, x2 != v, x2 != x1
            x2_mask = adj_bitmasks[x1] & ~(1 << u) & ~(1 << v) & ~(1 << x1)
            x2_candidates = mask_to_set(x2_mask)
            
            for x2 in x2_candidates:
                # x3 in adj[x2], x3 != u, x3 != v, x3 != x1, x3 != x2
                x3_mask = adj_bitmasks[x2] & ~(1 << u) & ~(1 << v) & ~(1 << x1) & ~(1 << x2)
                x3_candidates = mask_to_set(x3_mask)
                
                for x3 in x3_candidates:
                    # x4 in adj[x3], x4 != u, x4 != v, x4 != x1, x4 != x2, x4 != x3
                    # And v must be in adj[x4]
                    x4_mask = adj_bitmasks[x3] & ~(1 << u) & ~(1 << v) & ~(1 << x1) & ~(1 << x2) & ~(1 << x3)
                    # Also x4 must be adjacent to v
                    x4_mask &= adj_bitmasks[v]
                    
                    if x4_mask:
                        return True
        
        return False

    def solve_n(target_n):
        if time.time() > deadline:
            return None
        
        n = target_n
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Use bitmask adjacency for each color
        # adj[c] is a list of n bitmasks, adj[c][i] has bit j set if edge (i,j) has color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * m
        
        def backtrack(idx, color_order):
            if time.time() > deadline:
                return None
            
            if idx == m:
                return colors[:]
            
            u, v = edges[idx]
            
            # Try colors in given order
            for c in color_order:
                # Check if adding (u,v) with color c creates a C6
                # Temporarily add the edge
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                # Check for C6: is there a path of length 5 between u and v in color c?
                # Wait - the edge (u,v) is now in adj[c]. A C6 would be a cycle of length 6.
                # If we add (u,v), we need to check if there was already a path of length 5 from u to v.
                # But now (u,v) is in the graph. The C6 is: u - v - x1 - x2 - x3 - x4 - u? No.
                # Actually, adding edge (u,v) creates a C6 iff there exists a path of length 5 from u to v 
                # in the graph WITHOUT the edge (u,v).
                # So I should check BEFORE adding, or remove temporarily.
                
                # Let me undo and check properly
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                
                # Check if path of length 5 exists from u to v in current adj[c] (without edge u-v)
                if has_path_len_5(adj[c], u, v, n):
                    continue
                
                # Now add the edge
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                colors[idx] = c
                
                res = backtrack(idx + 1, color_order)
                if res is not None:
                    return res
                
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = -1
            
            return None
        
        # Symmetry breaking: Fix first few edges
        # Edge (0,1) -> color 0
        # Edge (0,2) -> color 0 or 1 (to break color permutation symmetry partially)
        
        # For now, just try all 4 colors in order
        # Could add more symmetry breaking later
        
        return backtrack(0, [0, 1, 2, 3])

    # Try n from 17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        res = solve_n(n)
        if res is not None:
            return {'n': n, 'colors': res}
    
    # Fallback to n=9
    n = 9
    # Cyclic coloring: color(i,j) = min(j-i, n-j+i) - 1
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            c = min(d, n - d) - 1
            colors.append(c)
    return {'n': n, 'colors': colors}
