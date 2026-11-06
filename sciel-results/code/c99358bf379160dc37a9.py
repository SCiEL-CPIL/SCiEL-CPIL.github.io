"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using bit-parallel backtracking search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for safety (150s limit)
    
    # Fallback solution: n=9 circular coloring
    def get_fallback(n):
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    best_n = 9
    best_colors = get_fallback(9)
    
    def solve_for_n(n, time_limit):
        """
        Attempts to find a 4-coloring of K_n with no monochromatic C6.
        Returns the list of colors if found, else None.
        """
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        
        # Map edge index to (u, v)
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        # Adjacency masks for each color: adj[color][vertex] is a bitmask of neighbors
        adj = [[0] * n for _ in range(4)]
        
        # Precompute edge indices for symmetry breaking
        def edge_index(u, v):
            # Number of edges before vertex u: u*(n-1) - u*(u-1)//2
            # Within vertex u's edges, v is the (v-u-1)th edge
            return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

        # Symmetry breaking:
        # Fix colors of edges from vertex 0 to break rotational/color symmetry.
        # This is a standard heuristic to reduce the search space by a factor of 4! or more.
        # We fix the first few edges of vertex 0 to specific colors.
        
        initial_fixes = []
        if n >= 2: initial_fixes.append((0, 1, 0))
        if n >= 3: initial_fixes.append((0, 2, 0))
        if n >= 4: initial_fixes.append((0, 3, 1))
        if n >= 5: initial_fixes.append((0, 4, 2))
        if n >= 6: initial_fixes.append((0, 5, 3))
        
        # Apply initial fixes
        for u, v, c in initial_fixes:
            idx = edge_index(u, v)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
        # Order edges for backtracking
        # We process edges in lexicographic order, but skip the fixed ones.
        unassigned_edges = []
        for idx in range(num_edges):
            if colors[idx] == -1:
                unassigned_edges.append(idx)
                
        # Heuristic: Process edges involving vertex 0 first? 
        # Actually, lexicographic order naturally prioritizes edges with smaller i.
        # Let's stick to that.
        
        def has_c6_with_new_edge(c, u, v):
            """
            Check if adding edge (u,v) to color c creates a C6.
            This is equivalent to checking if there is a path of length 5 between u and v
            in the graph G_c \ {(u,v)}.
            Path: u - a - b - c - d - v
            """
            mask_u = adj[c][u]
            mask_v = adj[c][v]
            
            # Iterate over 'a' in N(u)
            a = 0
            while mask_u:
                lsb = mask_u & -mask_u
                a = lsb.bit_length() - 1
                mask_u ^= lsb
                
                if a == v:
                    continue
                
                mask_a = adj[c][a]
                # b in N(a) \ {u, v}
                candidates_b = mask_a & ~(1 << u) & ~(1 << v)
                
                b = 0
                while candidates_b:
                    lsb_b = candidates_b & -candidates_b
                    b = lsb_b.bit_length() - 1
                    candidates_b ^= lsb_b
                    
                    mask_b = adj[c][b]
                    # c in N(b) \ {u, a, v}
                    candidates_c = mask_b & ~(1 << u) & ~(1 << a) & ~(1 << v)
                    
                    c_node = 0
                    while candidates_c:
                        lsb_c = candidates_c & -candidates_c
                        c_node = lsb_c.bit_length() - 1
                        candidates_c ^= lsb_c
                        
                        mask_c = adj[c][c_node]
                        # d in N(c) \ {u, a, b, v}
                        candidates_d = mask_c & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                        
                        d_node = 0
                        while candidates_d:
                            lsb_d = candidates_d & -candidates_d
                            d_node = lsb_d.bit_length() - 1
                            candidates_d ^= lsb_d
                            
                            mask_d = adj[c][d_node]
                            # Check if d is connected to v
                            if mask_d & (1 << v):
                                return True
            return False

        def backtrack(idx):
            """
            idx is the index in unassigned_edges list.
            """
            if time.time() > time_limit:
                return False
            
            if idx == len(unassigned_edges):
                return True
            
            edge_idx = unassigned_edges[idx]
            u, v = edge_map[edge_idx]
            
            # Try each color
            for c in range(4):
                # Check if this color is allowed
                if has_c6_with_new_edge(c, u, v):
                    continue
                
                # Assign
                colors[edge_idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[edge_idx] = -1
                
            return False

        # Try to solve
        if backtrack(0):
            return colors
        
        return None

    # Try to solve for n=17, 16, etc.
    # We use a deterministic approach with symmetry breaking.
    # If we fail, we might try random restarts, but symmetry breaking is usually strong enough.
    
    for target_n in [17, 16, 15, 14]:
        if time.time() > deadline:
            break
            
        # Try a few times with slightly different randomizations if we want,
        # but for now, deterministic with fixed symmetry break.
        res = solve_for_n(target_n, deadline)
        if res is not None:
            best_n = target_n
            best_colors = res
            break
            
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
