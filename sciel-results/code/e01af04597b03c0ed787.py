import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    def has_c6(adj, n):
        """
        Checks if the graph defined by adjacency bitsets 'adj' contains a 6-cycle.
        Returns a list of vertices forming the cycle, or None.
        adj[i] is an integer where bit j is set if edge (i, j) exists.
        """
        # Iterate over all pairs (v0, v1) such that edge exists
        for v0 in range(n):
            # Get neighbors of v0
            neigh_v0 = adj[v0]
            # Iterate over v1 in neighbors of v0
            while neigh_v0:
                v1 = (neigh_v0 & -neigh_v0).bit_length() - 1
                neigh_v0 &= neigh_v0 - 1
                
                # Ensure v0 < v1 to avoid duplicates? No, we just need any cycle.
                # But to avoid finding the same cycle multiple times or backtracking,
                # we can enforce an ordering or just accept any.
                
                # We need a path v1 -> v2 -> v3 -> v4 -> v5 -> v0
                # v2 must be neighbor of v1, not v0
                # v3 must be neighbor of v2, not v0, v1
                # v4 must be neighbor of v3, not v0, v1, v2
                # v5 must be neighbor of v4, not v0, v1, v2, v3
                # v0 must be neighbor of v5
                
                # Mask to exclude v0
                mask_v0 = ~(1 << v0)
                
                # Neighbors of v1 excluding v0
                neigh_v1 = adj[v1] & mask_v0
                
                while neigh_v1:
                    v2 = (neigh_v1 & -neigh_v1).bit_length() - 1
                    neigh_v1 &= neigh_v1 - 1
                    
                    # Neighbors of v2 excluding v0, v1
                    mask_v0_v1 = mask_v0 & ~(1 << v1)
                    neigh_v2 = adj[v2] & mask_v0_v1
                    
                    while neigh_v2:
                        v3 = (neigh_v2 & -neigh_v2).bit_length() - 1
                        neigh_v2 &= neigh_v2 - 1
                        
                        # Neighbors of v3 excluding v0, v1, v2
                        mask_v0_v1_v2 = mask_v0_v1 & ~(1 << v2)
                        neigh_v3 = adj[v3] & mask_v0_v1_v2
                        
                        while neigh_v3:
                            v4 = (neigh_v3 & -neigh_v3).bit_length() - 1
                            neigh_v3 &= neigh_v3 - 1
                            
                            # Neighbors of v4 excluding v0, v1, v2, v3
                            mask_v0_v1_v2_v3 = mask_v0_v1_v2 & ~(1 << v3)
                            neigh_v4 = adj[v4] & mask_v0_v1_v2_v3
                            
                            while neigh_v4:
                                v5 = (neigh_v4 & -neigh_v4).bit_length() - 1
                                neigh_v4 &= neigh_v4 - 1
                                
                                # Check if v0 is neighbor of v5
                                if (adj[v5] >> v0) & 1:
                                    return [v0, v1, v2, v3, v4, v5]
        return None

    def try_find_coloring(n, max_time):
        """
        Attempts to find a valid 4-coloring of K_n.
        Returns a list of colors or None.
        """
        if time.time() > max_time:
            return None
            
        # Initialize colors randomly
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Adjacency bitsets for each color
        # adj[c][i] is an int with bit j set if edge (i,j) has color c
        adj = [[0] * n for _ in range(4)]
        
        def update_adj(i, j, old_c, new_c):
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
        # Initialize adj
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i * (n - 1) - i * (i + 1) // 2 + j - i - 1] 
                # Wait, the index calculation in the prompt is lexicographic order.
                # Index k corresponds to pair (i, j) where:
                # k = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                # Let's verify:
                # i=0, j=1 -> 0 + 0 = 0. Correct.
                # i=0, j=2 -> 0 + 1 = 1. Correct.
                # i=1, j=2 -> 1*(n-1) - 1 + 0 = n-2.
                # For n=3, edges: (0,1), (0,2), (1,2). Indices 0, 1, 2.
                # i=1, j=2: 1*2 - 1 + 0 = 1? No.
                # Let's use a helper to get index.
                pass
        
        # Simpler index calculation
        def get_idx(i, j):
            # i < j
            # Number of edges starting with 0: n-1
            # Number of edges starting with 1: n-2
            # ...
            # Number of edges starting with i: n-1-i
            # So index is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
            # Sum is i*(n-1) - i*(i-1)//2
            # Total = i*(n-1) - i*(i-1)//2 + j - i - 1
            return i * (n - 1) - i * (i - 1) // 2 + j - i - 1

        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

        best_colors = colors[:]
        
        while time.time() < max_time:
            # Check for violations
            violation_found = False
            for c in range(4):
                cycle = has_c6(adj[c], n)
                if cycle:
                    violation_found = True
                    # Pick a random edge in the cycle and flip it
                    # Cycle: v0-v1-v2-v3-v4-v5-v0
                    # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
                    # We pick one edge, say (v0, v1)
                    v0, v1 = cycle[0], cycle[1]
                    if v0 > v1: v0, v1 = v1, v0
                    idx = get_idx(v0, v1)
                    old_c = c
                    new_c = random.choice([x for x in range(4) if x != old_c])
                    colors[idx] = new_c
                    update_adj(v0, v1, old_c, new_c)
                    break
            
            if not violation_found:
                return colors
                
            # If we have been stuck for too long, restart?
            # For now, just keep going.
            
        return None

    # Try n=17 first
    max_time_for_17 = 60
    result_17 = try_find_coloring(17, start_time + max_time_for_17)
    if result_17:
        # Verify result_17 is valid (just in case)
        # Re-construct adj and check
        n = 17
        adj = [[0] * n for _ in range(4)]
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j)
                c = result_17[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        valid = True
        for c in range(4):
            if has_c6(adj[c], n):
                valid = False
                break
        
        if valid:
            return {'n': 17, 'colors': result_17}

    # Try n=18 if time remains
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            result_18 = try_find_coloring(18, deadline)
            if result_18:
                n = 18
                adj = [[0] * n for _ in range(4)]
                def get_idx_18(i, j):
                    return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
                for i in range(n):
                    for j in range(i + 1, n):
                        idx = get_idx_18(i, j)
                        c = result_18[idx]
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                
                valid = True
                for c in range(4):
                    if has_c6(adj[c], n):
                        valid = False
                        break
                
                if valid:
                    return {'n': 18, 'colors': result_18}

    # Fallback to n=9 if we fail to find 17 or 18
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
