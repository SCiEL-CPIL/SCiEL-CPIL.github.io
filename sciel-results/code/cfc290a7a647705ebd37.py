import time
import random
import sys

# Set recursion limit higher just in case, though n=17 is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to be safe within 120s
    
    # Seed solution
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    
    def check_c6(n, colors, edge_idx, u, v, c):
        """
        Check if adding edge (u,v) with color c creates a C6 in color c.
        colors is a list of length n(n-1)/2.
        We maintain adjacency masks for each color.
        """
        # This function is called inside the search, so we need access to the current state.
        # It's better to pass the adjacency masks.
        pass

    # We will implement the search with bitset adjacency.
    # adj[color][vertex] = bitmask of neighbors in that color.
    
    def solve(n):
        num_edges = n * (n - 1) // 2
        # Precompute edge endpoints
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # State:
        # adj: list of 4 lists, each of length n, containing bitmask of neighbors
        adj = [[0] * n for _ in range(4)]
        colors = [0] * num_edges
        
        # Symmetry breaking:
        # We iterate through edges.
        # For edges incident to 0, we restrict colors based on previous edges incident to 0.
        
        def get_edge_idx(i, j):
            # i < j
            # index = i*n - i*(i+1)/2 + (j - i - 1)
            # Simpler: precomputed map
            return edge_map[(i, j)]
        
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def has_c6_path(u, v, c, adj):
            """
            Check if there is a path of length 5 from u to v in color c.
            If so, adding (u,v) creates a C6.
            """
            # BFS/DP up to depth 5
            # curr_reachable = {u}
            # next_reachable = neighbors(u)
            # ...
            # We use bitmasks.
            curr = 1 << u
            # We want to reach v in exactly 5 steps?
            # A path of length 5 has 6 vertices: v0, v1, v2, v3, v4, v5.
            # If we add edge (u,v), and there is a path u-v1-v2-v3-v4-v5-v, that's a cycle of length 6.
            # So we check if v is reachable from u in exactly 5 steps?
            # No, the path must be simple. But if there is a walk of length 5, does it imply a C6?
            # If the walk repeats a vertex, it contains a smaller cycle.
            # If it contains a C3, C4, or C5, that doesn't forbid C6.
            # However, if a walk of length 5 exists, it might not be a simple path.
            # But for sparse graphs, it's likely.
            # To be safe, we should check for simple paths.
            # Given n=17, we can just do a DFS for path of length 5.
            # But that's slow.
            # Let's use the property: if (A^5)_{u,v} > 0, there is a walk.
            # In random graphs, walks are likely to be simple or contain small cycles.
            # If it contains a small cycle, say C3, then we have a C3.
            # Does the presence of a C3 in color c matter? No, only C6 is forbidden.
            # So a walk of length 5 might not imply a C6.
            # Example: u-a-b-a-c-v. This is a walk of length 5. It contains a C4 (u-a-b-a? no, a-b-a is backtracking).
            # Actually, if we allow backtracking, it's not a simple path.
            # We need a simple path.
            # Let's implement a quick DFS for path of length 5.
            # n=17, depth 5. Branching factor ~3-4. 4^5 = 1024. Very fast.
            
            def dfs(current, target, depth, visited, adj_c):
                if depth == 5:
                    return (1 << target) & adj_c[current] != 0
                if depth > 5:
                    return False
                
                # Pruning: if target is not reachable in remaining steps, return False?
                # Simple: just iterate neighbors
                neighbors = adj_c[current]
                # Iterate bits
                while neighbors:
                    lsb = neighbors & -neighbors
                    nb = lsb.bit_length() - 1
                    if not (visited & lsb):
                        if dfs(nb, target, depth + 1, visited | lsb, adj_c):
                            return True
                    neighbors ^= lsb
                return False
            
            return dfs(u, v, 0, 1 << u, adj[c])

        def backtrack(edge_idx):
            if time.time() > deadline:
                return False
            
            if edge_idx == num_edges:
                return True
            
            u, v = edges[edge_idx]
            
            # Determine allowed colors
            # Symmetry breaking for edges incident to 0
            if u == 0:
                # Count distinct colors used on edges (0, k) for k < v
                # Actually, we just need to ensure we don't use a "new" color if an "old" one is available,
                # to reduce symmetry.
                # Standard trick: The color of (0, v) must be <= max_color_used_on_0_prev + 1?
                # Or simply: if we have used color 0 on (0, k1), we can use 0 or 1 on (0, v).
                # Let's track the set of colors used on edges (0, k) for k < v.
                used_colors = set()
                for k in range(1, v):
                    # Find color of (0, k)
                    # It's already assigned because k < v, so edge (0,k) comes before (0,v) in lexicographic order?
                    # Edges are (0,1), (0,2), ..., (0, n-1), (1,2)...
                    # Yes, (0,k) is assigned before (0,v) if k < v.
                    idx_k = edge_map[(0, k)]
                    used_colors.add(colors[idx_k])
                
                # We can use any color in {0, 1, 2, 3}.
                # But to break symmetry, if we haven't used color 1 yet, we can use 1.
                # If we have used 0 and 1, we can use 2.
                # Generally, we can use colors 0..max(used)+1, but capped at 3.
                # Actually, we can just try all 4, but order them: 0, 1, 2, 3.
                # The symmetry breaking is that we don't need to distinguish between color labels if they are symmetric.
                # By fixing the first edge to 0, we break the global color permutation.
                # By fixing subsequent edges to 0 or 1, we break more.
                
                # Let's just try 0, 1, 2, 3 in order. The first edge (0,1) is forced to 0.
                if v == 1:
                    allowed = [0]
                else:
                    # If we have used 0, we can use 0 or 1.
                    # If we have used 0, 1, we can use 0, 1, 2.
                    max_used = -1
                    for c in used_colors:
                        if c > max_used:
                            max_used = c
                    # We can use colors 0 to max_used + 1
                    allowed = list(range(min(max_used + 2, 4)))
            else:
                allowed = [0, 1, 2, 3]
            
            for c in allowed:
                # Check if adding (u, v) with color c creates a C6
                if not has_c6_path(u, v, c, adj):
                    # Assign
                    colors[edge_idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(edge_idx + 1):
                        return True
                    
                    # Unassign
                    adj[c][u] ^= (1 << v)
                    adj[c][v] ^= (1 << u)
            
            return False
        
        # Start search
        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    # Try to find solution for n=17, 16, ...
    # We prioritize higher n.
    # Since 17 is the best known, we try 17 first.
    # If it fails within deadline, we might try 16, but 16 is easier.
    # However, if 17 exists, we want it.
    
    best_result = None
    
    # Try 17
    res = solve(17)
    if res:
        return res
    
    # If 17 failed (or timed out), try 16
    # But solve(17) might have consumed most of the time.
    # We should check time.
    if time.time() < deadline - 10:
        res = solve(16)
        if res:
            return res
            
    if time.time() < deadline - 10:
        res = solve(15)
        if res:
            return res

    # Fallback to seed
    return {'n': seed_n, 'colors': seed_colors}

# EVOLVE-BLOCK-END
