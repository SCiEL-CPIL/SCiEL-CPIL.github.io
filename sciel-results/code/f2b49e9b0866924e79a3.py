def construct():
    import random
    import sys

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate edges
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []

        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build adjacency list for original graph
        adj_orig = [[] for _ in range(n)]
        for u, v in edges:
            adj_orig[u].append(v)
            adj_orig[v].append(u)
        
        # Precompute adjacency bitmasks for original graph vertices
        # adj_mask[v] is a bitmask of neighbors of v in G_orig
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # Build Conflict Graph
        # Two edges i, j conflict if:
        # 1. They share a vertex.
        # 2. They are joined by an edge in G_orig (i.e., there is an edge in G_orig connecting a vertex of i to a vertex of j).
        # Let edges[i] = (u1, v1), edges[j] = (u2, v2).
        # Conflict if:
        # (u1==u2 or u1==v2 or v1==u2 or v1==v2)  [Share vertex]
        # OR (u2 in adj_mask[u1] or v2 in adj_mask[u1] or u2 in adj_mask[v1] or v2 in adj_mask[v1]) [Joined]
        
        # We store conflict masks for each edge index i
        conflict = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            # Vertices involved in edge i
            verts_i = (u1, v1)
            
            # Neighbors of u1 and v1 in G_orig
            # We can use the precomputed adj_mask
            neigh_u1 = adj_mask[u1]
            neigh_v1 = adj_mask[v1]
            
            # For each other edge j, check conflict
            # Optimization: Iterate j and check conditions
            # Since m is small (< 300), O(m^2) is fine.
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                # Check 1: Share vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    # Check 2: Joined by edge
                    # Is u2 neighbor of u1? Is v2 neighbor of u1?
                    # Is u2 neighbor of v1? Is v2 neighbor of v1?
                    if (neigh_u1 >> u2) & 1 or (neigh_u1 >> v2) & 1 or (neigh_v1 >> u2) & 1 or (neigh_v1 >> v2) & 1:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        # MIS Solver on Conflict Graph
        # We want to find Max Independent Set in G_conf.
        # Independent Set in G_conf = Induced Matching in G_orig.
        
        sys.setrecursionlimit(10000)
        
        # Helper: Greedy Independent Set on a bitmask of candidates
        # Returns a lower bound on the size of the MIS of the subgraph induced by 'candidates'
        def greedy_is_size(candidates):
            count = 0
            while candidates:
                # Pick a vertex with minimum degree in the remaining graph?
                # For speed, just pick the first available bit.
                # A better heuristic: pick the vertex with the fewest neighbors in 'candidates'
                # to maximize the chance of adding more vertices.
                # Let's stick to first bit for speed in the bound calculation, 
                # but maybe improve pivot selection in B&B.
                v_bit = candidates & -candidates
                v = v_bit.bit_length() - 1
                count += 1
                # Remove v and its neighbors from candidates
                candidates &= ~(conflict[v] | v_bit)
            return count

        # Helper: Find the vertex in 'candidates' with the highest degree within 'candidates'
        def get_pivot(candidates):
            # This is expensive if done naively for every node.
            # m is small, so O(m) scan is okay.
            max_deg = -1
            pivot = -1
            # Iterate through bits in candidates
            c = candidates
            while c:
                bit = c & -c
                v = bit.bit_length() - 1
                # Degree of v in the subgraph induced by candidates
                deg = (conflict[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                c &= c - 1
            return pivot

        def bb_search(candidates, current_size, current_mask):
            nonlocal best_mis_size, best_mis_mask
            
            # Pruning
            if current_size + candidates.bit_count() <= best_mis_size:
                return
            
            if candidates == 0:
                if current_size > best_mis_size:
                    best_mis_size = current_size
                    best_mis_mask = current_mask
                return
            
            # Heuristic Bound: Greedy IS on remaining candidates
            # This provides an upper bound on how many more we can add.
            # If current_size + greedy_bound <= best, prune.
            # Note: Greedy IS size is a LOWER bound on MIS size? No, it's an approximation.
            # Wait. The size of a greedy independent set is a LOWER bound on the MIS size?
            # No, MIS >= Greedy IS. So if we find a set of size K, MIS is at least K.
            # We want to prune if current_size + (Max Possible Additional) <= best.
            # Max Possible Additional is bounded by MIS(candidates).
            # Greedy IS gives a valid IS, so MIS(candidates) >= Greedy_IS(candidates).
            # This doesn't help prune directly unless we have an UPPER bound.
            # A simple upper bound is candidates.bit_count().
            # A better upper bound: Partition candidates into cliques? Hard.
            # Let's use a different bound: 
            # If we pick a pivot v, any IS either contains v (then no neighbors of v) or does not.
            # 
            # Let's try a simple greedy UPPER bound heuristic:
            # Actually, a common trick is to use the greedy IS size as a lower bound for the *remaining* search, 
            # but we need an upper bound to prune.
            # 
            # Let's rely on the pivot branching.
            
            # Select Pivot
            pivot = get_pivot(candidates)
            if pivot == -1:
                return
                
            pivot_bit = 1 << pivot
            
            # Branch 1: Include pivot
            # New candidates: candidates without pivot and without its neighbors
            new_candidates = candidates & ~(conflict[pivot] | pivot_bit)
            bb_search(new_candidates, current_size + 1, current_mask | pivot_bit)
            
            # Branch 2: Exclude pivot
            new_candidates = candidates & ~pivot_bit
            bb_search(new_candidates, current_size, current_mask)

        best_mis_size = 0
        best_mis_mask = 0
        
        # Initial Greedy to set a good lower bound
        # Use a simple greedy on the full graph
        c = (1 << m) - 1
        init_mask = 0
        init_size = 0
        while c:
            # Pick vertex with min degree in remaining? 
            # Let's just pick first bit for initialization speed
            v_bit = c & -c
            v = v_bit.bit_length() - 1
            init_size += 1
            init_mask |= v_bit
            c &= ~(conflict[v] | v_bit)
            
        best_mis_size = init_size
        best_mis_mask = init_mask

        bb_search((1 << m) - 1, 0, 0)
        
        # Now we have the largest induced matching (MIS of conflict graph).
        # We assign color 0 to these edges.
        # Then remove them and repeat.
        
        colors = [0] * m
        available_mask = (1 << m) - 1
        color_idx = 0
        
        while available_mask:
            # Find MIS in subgraph induced by available_mask
            # Reset global bests for this sub-instance
            
            local_best_size = 0
            local_best_mask = 0
            
            # We need to run BB again on the subgraph.
            # To avoid re-defining the function with nonlocals every time, 
            # we can pass the context or just re-define it inside the loop.
            # Re-defining is fine as m is small.
            
            def bb_search_subset(candidates, current_size, current_mask):
                nonlocal local_best_size, local_best_mask
                
                if current_size + candidates.bit_count() <= local_best_size:
                    return
                
                if candidates == 0:
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_mask = current_mask
                    return
                
                # Pruning with a simple upper bound:
                # The number of vertices in candidates is an upper bound on the IS size.
                # We already check that.
                
                # Select Pivot
                pivot = get_pivot(candidates)
                if pivot == -1:
                    return
                    
                pivot_bit = 1 << pivot
                
                # Branch 1: Include pivot
                new_candidates = candidates & ~(conflict[pivot] | pivot_bit)
                bb_search_subset(new_candidates, current_size + 1, current_mask | pivot_bit)
                
                # Branch 2: Exclude pivot
                new_candidates = candidates & ~pivot_bit
                bb_search_subset(new_candidates, current_size, current_mask)

            # Initialize local best with a greedy on available_mask
            c = available_mask
            init_mask = 0
            init_size = 0
            while c:
                v_bit = c & -c
                v = v_bit.bit_length() - 1
                init_size += 1
                init_mask |= v_bit
                c &= ~(conflict[v] | v_bit)
                
            local_best_size = init_size
            local_best_mask = init_mask

            bb_search_subset(available_mask, 0, 0)
            
            # Assign color
            mis_mask = local_best_mask
            # Iterate through bits in mis_mask
            temp_mask = mis_mask
            while temp_mask:
                bit = temp_mask & -temp_mask
                idx = bit.bit_length() - 1
                colors[idx] = color_idx
                available_mask &= ~bit
                temp_mask &= ~bit
            
            color_idx += 1
            
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        res = solve_instance(n, d, seed)
        out.append(res)
        
    return out
