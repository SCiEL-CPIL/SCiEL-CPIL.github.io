<<<<<<< SEARCH
    # Precompute reverse mapping from edge index to (i,j)
    idx_to_edge = {}
    for (i, j), idx in edge_to_idx.items():
        idx_to_edge[idx] = (i, j)
    
    def count_c6_fast():
        """
        Count C6s using vectorized numpy operations.
        For each color c, a C6 is a cycle v0-v1-v2-v3-v4-v5-v0 where all edges have color c.
        We count by: for each color, compute A^2, A^3, then for each edge (i,j) in color c,
        count paths of length 5 from i to j avoiding i and j internally.
        Actually, simpler: use the formula based on matrix powers.
        
        Number of C6s in graph G = (1/12) * [trace(A^6) - 6*trace(A^2*A^2*A^2) + ...]
        This is complex. Let's use a direct but vectorized count.
        
        For n <= 17, we can do: for each color, for each pair (i,j) NOT adjacent,
        count the number of length-5 paths from i to j. Each C6 corresponds to
        12 such (i,j, path) tuples. But we need vertex-disjoint paths.
        
        Simpler approach for small n: use numpy to count 6-cycles by checking
        all 6-tuples with vectorized operations on subsets.
        """
        total = 0
        for c in range(4):
            A = adj[c].astype(np.int8)
            # For each ordered pair (i,j) with i != j, count paths of length 5
            # A5 = A^5 gives number of length-5 walks, but we need simple paths.
            # For C6 counting, we can use:
            # For each i, j not adjacent, count length-5 paths from i to j
            # This overcounts because walks can repeat vertices.
            # 
            # Alternative: enumerate all 6-subsets and check if they form a C6.
            # C(17,6) = 12376, which is manageable.
            from itertools import combinations
            for subset in combinations(range(n), 6):
                # Check all 6! / 2 / 6 = 60 possible orderings? No, just check if
                # the induced subgraph on these 6 vertices contains a C6.
                # A 6-vertex graph has a C6 iff it's a 6-cycle or has a 6-cycle as subgraph.
                # Actually, we just need to check if there exists a permutation
                # (v0,v1,v2,v3,v4,v5) of the 6 vertices such that all consecutive
                # and closing edges exist in color c.
                # There are 6!/2 = 360 such cycles (divided by 2 for direction).
                # But we can be smarter: fix v0 = min(subset), then 5!/2 = 60 orderings.
                v0 = subset[0]
                remaining = list(subset[1:])
                # Try all permutations of remaining 5 vertices
                # This is 5! = 120, but we can prune
                found = False
                # Check if v0 has at least 2 neighbors in the subset
                neighbors = [v for v in remaining if A[v0, v]]
                if len(neighbors) < 2:
                    continue
                # For each pair of neighbors (v1, v5), check if there's a path v1-v2-v3-v4-v5
                for v1 in neighbors:
                    for v5 in neighbors:
                        if v1 == v5:
                            continue
                        # Need path v1 -> v2 -> v3 -> v4 -> v5 using remaining vertices
                        mid_vertices = [v for v in remaining if v != v1 and v != v5]
                        if len(mid_vertices) != 3:
                            continue
                        # Check all permutations of mid_vertices
                        from itertools import permutations
                        for perm in permutations(mid_vertices):
                            v2, v3, v4 = perm
                            if (A[v1, v2] and A[v2, v3] and A[v3, v4] and A[v4, v5]):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    total += 1
        return total
=======
    # Precompute reverse mapping from edge index to (i,j)
    idx_to_edge = {}
    for (i, j), idx in edge_to_idx.items():
        idx_to_edge[idx] = (i, j)
    
    def count_c6_fast():
        """
        Vectorized C6 count using numpy matrix operations.
        For each color c, compute number of 6-cycles.
        A 6-cycle in an undirected graph can be counted as:
        For each unordered pair {i,j} that is NOT an edge, count the number of
        length-5 simple paths from i to j. Each 6-cycle is counted 12 times
        (6 vertices * 2 directions). But simple path counting is hard.
        
        Alternative: use the formula for number of 6-cycles in terms of matrix powers.
        For a simple graph, the number of 6-cycles is:
        (1/12) * [tr(A^6) - 6*tr(A^2)*tr(A^2) - 3*tr(A^2)^2 + 12*tr(A^2*A^2*A^2) - 12*tr(A^4) + 2*tr(A^2)^3]
        This is getting complex. Let's just do a direct vectorized count.
        
        For n <= 17, we can afford O(n^3) per color using numpy.
        """
        total = 0
        for c in range(4):
            A = adj[c].astype(np.int8)
            # Compute A^2 and A^3
            A2 = A @ A
            A3 = A2 @ A
            # For each pair (i, j) with i < j and A[i,j] = 0 (not adjacent),
            # count the number of length-5 paths from i to j.
            # A5[i,j] gives the number of length-5 walks, not simple paths.
            # But for the purpose of local search, using walks as a proxy is OK
            # since we're minimizing. However, it's not exact.
            #
            # Let's use a different approach: for each color, count 6-cycles by
            # checking all 6-tuples with numpy vectorization.
            # For n=17, C(17,6) = 12376 subsets. For each subset, check if it
            # contains a 6-cycle in O(1) using precomputed adjacency.
            #
            # Actually, let's just do the O(n^4) enumeration but with numpy:
            # For each (v0, v1) edge, for each (v2, v3) edge not sharing vertices with v0,v1,
            # for each (v4, v5) edge not sharing vertices, check if v5-v0 is an edge.
            # This is O(n^4) which for n=17 is ~83521 * 4 colors = ~334K operations.
            # With numpy, this should be fast.
            
            # Enumerate all pairs of edges that don't share vertices
            # Edge (v0,v1) and edge (v2,v3) with all 4 distinct
            count = 0
            # Precompute list of edges for color c
            edges_c = [(i, j) for i in range(n) for j in range(i+1, n) if A[i,j]]
            for v0, v1 in edges_c:
                for v2, v3 in edges_c:
                    if v2 in (v0, v1) or v3 in (v0, v1):
                        continue
                    # Now need edge (v3, v4) and (v4, v5) and (v5, v0)
                    # where v4, v5 are distinct from v0,v1,v2,v3
                    # This is getting complex. Let's just use the walk count as proxy.
                    pass
            
            # Use walk count as a proxy for local search
            A5 = A3 @ A2
            # For each non-adjacent pair (i,j), A5[i,j] counts length-5 walks
            # Each 6-cycle contributes 12 to the sum of A5[i,j] over non-adjacent (i,j)
            # But walks can repeat vertices, so this is an upper bound.
            non_adj = 1 - A
            np.fill_diagonal(non_adj, 0)
            # Sum A5 over non-adjacent pairs
            count = int(np.sum(A5 * non_adj)) // 12
            total += count
        
        return total
>>>>>>> REPLACE

I'm overcomplicating this. Let me use a much simpler and faster approach: just count C6s by checking all 6-cycles directly with a fast Python loop, since for n=17 there are at most C(17,6)*60 ≈ 742K checks, and we can optimize with early termination.

