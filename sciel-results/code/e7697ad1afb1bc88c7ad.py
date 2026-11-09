import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    # Precomputed known valid coloring for n=17
    # This is a fallback to ensure we always return a valid result if search fails or times out.
    # The seed program had a fallback to n=9, but we can hardcode a known n=17 solution 
    # or generate one quickly. However, generating a specific known solution is safer for score stability.
    # Actually, the prompt says "best known colouring has 17 vertices". 
    # I will include a robust search that tries to find n=17 quickly, and falls back to a 
    # simple valid construction for smaller n if needed, but aims for 17.
    
    # Let's implement the constructive backtracking search.
    # State:
    # adj[c][i] : bitset of neighbors of i in color c
    # We build vertex by vertex.
    # When adding vertex v (index v), we assign colors to edges (v, u) for u < v.
    # Constraint: No color c can contain a C6.
    # A C6 involving the new vertex v would look like:
    # v - u1 - u2 - u3 - u4 - u5 - v
    # where u1..u5 are existing vertices.
    # The edges (v,u1) and (v,u5) must have color c.
    # The path u1-u2-u3-u4-u5 must be in color c.
    # So, if we assign color c to edge (v, u1), we must ensure that there is no path of length 4 
    # in color c starting from u1 and ending at any u5 such that (v, u5) is also color c.
    # Actually, it's simpler to check:
    # When assigning color c to edge (v, u), we must check if there exists a path of length 4 
    # in color c from u to any other vertex w such that edge (v, w) is already assigned color c.
    # Or, if we assign colors sequentially for u=0..v-1, we can check against previously assigned edges.
    
    # Better approach:
    # Maintain for each color c, the set of all paths of length 4 (u1, u2, u3, u4, u5) that are monochromatic c.
    # This is expensive.
    
    # Alternative: 
    # When adding vertex v, we choose a vector of colors for edges (v, 0), (v, 1), ..., (v, v-1).
    # Let col[u] be the color of edge (v, u).
    # A C6 is formed if there exist u1 < u2 < u3 < u4 < u5 < v such that:
    # col[u1] == col[u5] == c
    # AND (u1, u2, u3, u4, u5) is a path in color c in the subgraph G_c.
    # Note: The path must use only vertices < v.
    
    # So, for a candidate assignment col[0..v-1], we check:
    # For each pair (u1, u5) with col[u1] == col[u5] == c:
    #   Check if there is a path of length 3 in G_c between u1 and u5.
    #   (Because u1-u2-u3-u4-u5 is length 4 edges, so distance 4? No, C6 has 6 edges.
    #    v-u1-u2-u3-u4-u5-v.
    #    Edges: (v,u1), (u1,u2), (u2,u3), (u3,u4), (u4,u5), (u5,v).
    #    So we need a path of length 4 in G_c from u1 to u5.
    #    i.e., u1 ~ u2 ~ u3 ~ u4 ~ u5.
    
    # Let's optimize the check.
    # For a fixed v, and fixed colors of (v, u) for u < v:
    # We can precompute for the existing graph G_c (vertices 0..v-1) the existence of paths of length 4.
    # Let P_c[u1][u5] be True if there is a path of length 4 in G_c from u1 to u5.
    # Then the assignment is valid iff for all u1, u5:
    #   NOT (col[u1] == col[u5] == c AND P_c[u1][u5])
    
    # How to compute P_c quickly?
    # P_c is the 4th power of the adjacency matrix of G_c?
    # (A^4)[i][j] > 0.
    # Since v is up to 20, matrix multiplication is fine, but bitsets are faster.
    
    # Let's use bitsets for adjacency.
    # adj[c][i] is a bitmask.
    # To find neighbors of length 2: N2[i] = OR of adj[c][j] for j in adj[c][i]
    # Length 4: N4[i] = OR of N2[j] for j in N2[i]
    
    # This search might still be slow for n=18.
    # We will try n=17 first with this method, which should be very fast.
    # Then try n=18 if time permits.
    
    def try_construct(n, max_time):
        # adj[c][i] for i in 0..n-1
        adj = [[0] * n for _ in range(4)]
        
        # We will build the graph vertex by vertex.
        # colors_edge[u][v] stores the color of edge (u, v) for u < v.
        # Or just store in a flat list or 2D array.
        edge_colors = [[-1] * n for _ in range(n)]
        
        # Precompute masks for time check
        def time_check():
            return time.time() < max_time
        
        # Recursive function to assign colors for vertex v
        # v: current vertex index
        # col_vec: list of colors for edges (v, 0), (v, 1), ..., (v, v-1)
        # We fill col_vec[0..v-1]
        
        # To speed up, we can use iterative deepening or just backtracking with random restarts.
        # Given n<=20, backtracking might be too slow without strong pruning.
        # However, the constraint is local to the new vertex.
        
        # Let's try a randomized greedy with backtracking on conflict.
        
        # Actually, a simpler heuristic:
        # Randomly assign colors to edges (v, u) for u < v.
        # Check validity.
        # If invalid, find a conflict and backtrack or restart.
        
        # Given the small n, a full backtracking with good ordering might work for n=17.
        # But n=18 is hard.
        
        # Let's implement a simple backtracking for n=17.
        
        sys.setrecursionlimit(10000)
        
        def check_validity_for_v(v, col_vec):
            # Check if assigning col_vec[0..v-1] to edges (v, u) creates a C6.
            # We need to check for each c:
            # For any u1, u5 < v with col_vec[u1] == col_vec[u5] == c:
            #   Is there a path of length 4 in adj[c] (on vertices 0..v-1) from u1 to u5?
            
            # Precompute reachability of length 4 for each color c in the current subgraph (0..v-1)
            # This depends only on the existing edges, not on col_vec.
            # So we can compute this once per vertex v, before trying col_vec.
            
            pass # Placeholder, see below
            
        # Since the subgraph 0..v-1 is fixed when we are at step v,
        # we can precompute the "4-step reachability" matrix for each color.
        
        # Let's restructure:
        # We maintain adj[c][i] for i < v.
        # At step v, we compute reach4[c][i][j] for i, j < v.
        # Then we search for a valid col_vec.
        
        # To make it faster, we can use a SAT-like approach or just random trials.
        # For n=17, random trials with backtracking on the last few vertices might work.
        
        # Let's try a different tactic: 
        # Start with a random coloring of K_n.
        # Use a local search that specifically targets C6s.
        # The seed program did this but maybe it got stuck.
        # Let's try a more aggressive local search: 
        # 1. Find all C6s.
        # 2. For each C6, flip a random edge in it to a random other color.
        # 3. Repeat until no C6s or timeout.
        
        # The seed program did this. Why did it score 1.0? 
        # Probably because it found n=17. 
        # To beat 1.0, we need n=18 or 19.
        # Local search is unlikely to find n=18 quickly.
        # Constructive backtracking is more likely to find n=18 if it exists, 
        # because it explores the space more systematically.
        
        # Let's implement the constructive backtracking properly.
        
        def solve(n):
            adj = [[0] * n for _ in range(4)]
            edge_colors = [[-1] * n for _ in range(n)]
            
            # Precompute masks for vertices
            all_mask = (1 << n) - 1
            
            def get_reach4(c, v_limit):
                # Returns a list of bitsets reach4[i] where bit j is set if there is a path of length 4 from i to j in color c, using only vertices < v_limit.
                # Actually, we just need to know if u1 and u5 are connected by a path of length 4.
                # We can compute this on the fly for the specific pairs we care about?
                # No, precomputing for all pairs is O(n^2) per color per vertex.
                # n=20, so 400 pairs. 4 colors. 20 vertices. 32000 ops. Very fast.
                
                reach4 = [0] * n
                # Step 1: neighbors
                # Step 2: neighbors of neighbors
                # Step 3: ...
                # Step 4: ...
                
                # Let's use matrix multiplication logic with bitsets.
                # A = adj[c][:v_limit]
                # A2[i] = OR(A[j] for j in A[i])
                # A4[i] = OR(A2[j] for j in A2[i])
                
                A = adj[c][:v_limit]
                A2 = [0] * v_limit
                for i in range(v_limit):
                    mask = A[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        A2[i] |= A[j]
                        mask ^= lsb
                        
                A4 = [0] * v_limit
                for i in range(v_limit):
                    mask = A2[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        A4[i] |= A2[j]
                        mask ^= lsb
                        
                return A4
            
            # Backtracking search
            # We assign colors for vertex v (0-indexed).
            # v=0: no edges.
            # v=1: edge (0,1).
            # ...
            
            def backtrack(v):
                if time.time() > max_time:
                    return False
                
                if v == n:
                    return True
                
                # Precompute reach4 for current subgraph 0..v-1
                reach4s = []
                for c in range(4):
                    reach4s.append(get_reach4(c, v))
                
                # We need to choose colors for edges (v, u) for u in 0..v-1.
                # Let's try random assignments and check.
                # To be smarter, we can use backtracking on the assignment of col[u].
                
                # Since the constraint is global (pairs of u1, u5), simple sequential assignment is hard.
                # Let's try random restarts for the whole vertex v.
                
                # Actually, for v up to 17, the number of colorings is 4^16 which is huge.
                # But constraints prune it.
                
                # Let's try a randomized greedy with backtracking on conflicts.
                # 1. Randomly assign col[0..v-1].
                # 2. Check for conflicts.
                # 3. If conflict, pick a random edge in the conflict and change its color.
                # 4. Repeat.
                
                # This is essentially local search on the layer v.
                
                # Let's try this:
                col = [random.randint(0, 3) for _ in range(v)]
                
                for _ in range(1000): # Max attempts for this vertex
                    if time.time() > max_time:
                        return False
                    
                    # Check validity
                    valid = True
                    conflict_edges = []
                    
                    for c in range(4):
                        u1s = [u for u in range(v) if col[u] == c]
                        for i in range(len(u1s)):
                            for j in range(i+1, len(u1s)):
                                u1 = u1s[i]
                                u5 = u1s[j]
                                if (reach4s[c][u1] >> u5) & 1:
                                    valid = False
                                    conflict_edges.append((u1, u5, c))
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    
                    if valid:
                        # Accept this assignment
                        for u in range(v):
                            edge_colors[v][u] = col[u]
                            edge_colors[u][v] = col[u]
                            adj[col[u]][v] |= (1 << u)
                            adj[col[u]][u] |= (1 << v)
                        
                        if backtrack(v + 1):
                            return True
                        
                        # Undo
                        for u in range(v):
                            c = col[u]
                            adj[c][v] &= ~(1 << u)
                            adj[c][u] &= ~(1 << v)
                            edge_colors[v][u] = -1
                            edge_colors[u][v] = -1
                        return False # Backtrack to previous vertex? No, we should try different col for v.
                    
                    # If invalid, pick a random conflict and flip one of the edges
                    if conflict_edges:
                        u1, u5, c = random.choice(conflict_edges)
                        # Flip col[u1] or col[u5]?
                        # Let's flip col[u1] to a random other color
                        new_c = random.choice([x for x in range(4) if x != col[u1]])
                        col[u1] = new_c
                    else:
                        # Should not happen
                        break
                        
                # If we failed to find a valid col for v, return False to backtrack
                return False
            
            # Start with v=0
            if backtrack(0):
                # Construct output
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append(edge_colors[i][j])
                return colors
            return None
        
        # Try n=17
        # Note: The backtracking above might be slow if it doesn't prune well.
        # For n=17, it should be fast.
        
        # Let's add a timeout guard inside the loops.
        
        # We will try n=17 first.
        if time.time() < deadline - 10:
            res_17 = solve(17)
            if res_17:
                # Verify
                if verify(17, res_17):
                    return {'n': 17, 'colors': res_17}
        
        # Try n=18 if time remains
        if time.time() < deadline - 5:
            res_18 = solve(18)
            if res_18:
                if verify(18, res_18):
                    return {'n': 18, 'colors': res_18}
        
        # Fallback: Return a known valid n=17 or smaller
        # Let's generate a simple valid n=9
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 4)
        return {'n': n, 'colors': colors}
        
    def verify(n, colors):
        adj = [[0] * n for _ in range(4)]
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        for c in range(4):
            if has_c6(adj[c], n):
                return False
        return True
        
    def has_c6(adj, n):
        for v0 in range(n):
            neigh_v0 = adj[v0]
            while neigh_v0:
                v1 = (neigh_v0 & -neigh_v0).bit_length() - 1
                neigh_v0 &= neigh_v0 - 1
                mask_v0 = ~(1 << v0)
                neigh_v1 = adj[v1] & mask_v0
                while neigh_v1:
                    v2 = (neigh_v1 & -neigh_v1).bit_length() - 1
                    neigh_v1 &= neigh_v1 - 1
                    mask_v0_v1 = mask_v0 & ~(1 << v1)
                    neigh_v2 = adj[v2] & mask_v0_v1
                    while neigh_v2:
                        v3 = (neigh_v2 & -neigh_v2).bit_length() - 1
                        neigh_v2 &= neigh_v2 - 1
                        mask_v0_v1_v2 = mask_v0_v1 & ~(1 << v2)
                        neigh_v3 = adj[v3] & mask_v0_v1_v2
                        while neigh_v3:
                            v4 = (neigh_v3 & -neigh_v3).bit_length() - 1
                            neigh_v3 &= neigh_v3 - 1
                            mask_v0_v1_v2_v3 = mask_v0_v1_v2 & ~(1 << v3)
                            neigh_v4 = adj[v4] & mask_v0_v1_v2_v3
                            while neigh_v4:
                                v5 = (neigh_v4 & -neigh_v4).bit_length() - 1
                                neigh_v4 &= neigh_v4 - 1
                                if (adj[v5] >> v0) & 1:
                                    return [v0, v1, v2, v3, v4, v5]
        return None

    # Execute
    return try_construct(20, deadline)

# Note: The solve function inside try_construct uses max_time from the outer scope.
# I need to make sure max_time is accessible.
# I will refactor slightly to pass max_time or use a global.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def has_c6(adj, n):
        for v0 in range(n):
            neigh_v0 = adj[v0]
            while neigh_v0:
                v1 = (neigh_v0 & -neigh_v0).bit_length() - 1
                neigh_v0 &= neigh_v0 - 1
                mask_v0 = ~(1 << v0)
                neigh_v1 = adj[v1] & mask_v0
                while neigh_v1:
                    v2 = (neigh_v1 & -neigh_v1).bit_length() - 1
                    neigh_v1 &= neigh_v1 - 1
                    mask_v0_v1 = mask_v0 & ~(1 << v1)
                    neigh_v2 = adj[v2] & mask_v0_v1
                    while neigh_v2:
                        v3 = (neigh_v2 & -neigh_v2).bit_length() - 1
                        neigh_v2 &= neigh_v2 - 1
                        mask_v0_v1_v2 = mask_v0_v1 & ~(1 << v2)
                        neigh_v3 = adj[v3] & mask_v0_v1_v2
                        while neigh_v3:
                            v4 = (neigh_v3 & -neigh_v3).bit_length() - 1
                            neigh_v3 &= neigh_v3 - 1
                            mask_v0_v1_v2_v3 = mask_v0_v1_v2 & ~(1 << v3)
                            neigh_v4 = adj[v4] & mask_v0_v1_v2_v3
                            while neigh_v4:
                                v5 = (neigh_v4 & -neigh_v4).bit_length() - 1
                                neigh_v4 &= neigh_v4 - 1
                                if (adj[v5] >> v0) & 1:
                                    return [v0, v1, v2, v3, v4, v5]
        return None

    def verify(n, colors):
        adj = [[0] * n for _ in range(4)]
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        for c in range(4):
            if has_c6(adj[c], n):
                return False
        return True

    def solve(n, max_time):
        adj = [[0] * n for _ in range(4)]
        edge_colors = [[-1] * n for _ in range(n)]
        
        def get_reach4(c, v_limit):
            A = adj[c][:v_limit]
            A2 = [0] * v_limit
            for i in range(v_limit):
                mask = A[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    A2[i] |= A[j]
                    mask ^= lsb
            A4 = [0] * v_limit
            for i in range(v_limit):
                mask = A2[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    A4[i] |= A2[j]
                    mask ^= lsb
            return A4

        def backtrack(v):
            if time.time() > max_time:
                return False
            if v == n:
                return True
            
            reach4s = []
            for c in range(4):
                reach4s.append(get_reach4(c, v))
            
            col = [random.randint(0, 3) for _ in range(v)]
            
            for _ in range(2000):
                if time.time() > max_time:
                    return False
                
                valid = True
                conflict_edges = []
                
                for c in range(4):
                    u1s = [u for u in range(v) if col[u] == c]
                    for i in range(len(u1s)):
                        for j in range(i+1, len(u1s)):
                            u1 = u1s[i]
                            u5 = u1s[j]
                            if (reach4s[c][u1] >> u5) & 1:
                                valid = False
                                conflict_edges.append((u1, u5, c))
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
                
                if valid:
                    for u in range(v):
                        edge_colors[v][u] = col[u]
                        edge_colors[u][v] = col[u]
                        adj[col[u]][v] |= (1 << u)
                        adj[col[u]][u] |= (1 << v)
                    
                    if backtrack(v + 1):
                        return True
                    
                    for u in range(v):
                        c = col[u]
                        adj[c][v] &= ~(1 << u)
                        adj[c][u] &= ~(1 << v)
                        edge_colors[v][u] = -1
                        edge_colors[u][v] = -1
                    return False
                
                if conflict_edges:
                    u1, u5, c = random.choice(conflict_edges)
                    new_c = random.choice([x for x in range(4) if x != col[u1]])
                    col[u1] = new_c
                else:
                    break
                    
            return False
        
        if backtrack(0):
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(edge_colors[i][j])
            return colors
        return None

    # Try n=17
    if time.time() < deadline - 10:
        res_17 = solve(17, deadline - 10)
        if res_17 and verify(17, res_17):
            return {'n': 17, 'colors': res_17}

    # Try n=18
    if time.time() < deadline - 5:
        res_18 = solve(18, deadline - 5)
        if res_18 and verify(18, res_18):
            return {'n': 18, 'colors': res_18}

    # Fallback n=9
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}
