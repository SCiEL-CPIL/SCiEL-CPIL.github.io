def construct():
    import time
    start = time.time()
    DEADLINE = 55  # seconds, safety margin before 60s
    
    N = 256
    # Build conflict graph: conflict[i] is a bitset of vertices that conflict with vertex i
    # Vertices are 1..256, index 0..255 for vertex 1..256
    
    # Two distinct vertices a,b conflict if (a+b) divides a*b
    conflict = [0] * N  # conflict[i] has bit j set if vertex (i+1) conflicts with vertex (j+1)
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Maximum independent set via branch and bound
    # Use bitset representation
    # Upper bound: greedy coloring on the conflict subgraph
    
    best = [0]  # best independent set size
    best_set = [0]  # best independent set as bitset
    
    # Order vertices by degree (higher degree first for pruning)
    degrees = [bin(conflict[i]).count('1') for i in range(N)]
    order = sorted(range(N), key=lambda x: -degrees[x])
    
    # Map: position in order -> vertex index
    # We'll work in the original vertex space but process in order
    
    def greedy_color_upper_bound(candidates_bitset):
        """Upper bound on independent set size via greedy coloring of conflict graph."""
        # Greedy coloring: assign colors to vertices in candidates
        # The number of colors is an upper bound on clique size, 
        # but for independent set we need: the number of colors in a proper coloring
        # of the complement gives upper bound... 
        # Actually: in the conflict graph, an independent set is a set with no edges.
        # Greedy coloring of the conflict graph: each color class is an independent set.
        # So the max independent set size >= max color class size.
        # That's a LOWER bound, not upper.
        
        # For upper bound: we can use the fact that if we partition into k independent sets,
        # then alpha <= ... no.
        
        # Better upper bound: maximum matching gives lower bound on vertex cover,
        # so alpha = n - tau <= n - (max matching size)
        # But that's expensive.
        
        # Simpler: use a greedy approach to get an upper bound.
        # Actually, let's use: the number of vertices in candidates is a trivial upper bound.
        # Better: use a simple greedy to find a large independent set (lower bound for pruning is not helpful).
        
        # For upper bound in branch and bound for MIS:
        # We can use the fact that if we greedily pick vertices and their neighbors,
        # the remaining structure gives bounds.
        
        # Simple approach: return the count of candidates as upper bound (weak but safe)
        return bin(candidates_bitset).count('1')
    
    # Better: use a proper upper bound based on coloring
    def coloring_upper_bound(candidates_bitset):
        """
        Upper bound on maximum independent set size.
        In the conflict graph, we want max set with no edges.
        Greedy coloring of the conflict graph: if we can color with k colors,
        each color class is an independent set, so alpha >= max class size (lower bound).
        
        For UPPER bound: we need a different approach.
        Actually, the standard approach for MIS branch and bound uses:
        - A lower bound (current best) for pruning
        - An upper bound: the number of remaining vertices is trivial
        - Better: use the fact that if we have a clique of size k in the conflict graph,
          we can pick at most 1 from it, so alpha <= n - k + 1... no that's not right either.
        
        Let me use a simpler but effective approach:
        - Process vertices in a fixed order
        - At each step, either include or exclude the vertex
        - Upper bound: current_size + (number of remaining candidates that don't conflict with current selection)
        """
        return bin(candidates_bitset).count('1')
    
    # Let me use a cleaner branch and bound:
    # State: (candidates_bitset, current_independent_set_bitset)
    # We process vertices in a fixed order (by degree)
    
    # Actually, let me use a well-known approach:
    # Branch on each vertex: include it (remove its neighbors from candidates) or exclude it
    # Upper bound: current_size + popcount(candidates)
    # This is weak. Better: use a greedy coloring upper bound.
    
    # For a proper upper bound, let's use: 
    # In the subgraph induced by candidates, compute a greedy coloring of the COMPLEMENT
    # graph. Each color class in the complement is a clique in the conflict graph.
    # From a clique of size k, we can pick at most 1 vertex.
    # So alpha <= sum over color classes of 1 = number of color classes? No...
    # 
    # Actually: if we color the conflict graph with k colors, each color class is independent.
    # That doesn't directly give an upper bound on alpha.
    #
    # The standard MIS branch and bound uses:
    # - Lower bound: best solution found so far
    # - Upper bound: |candidates| + current_size (trivial)
    # - Or: use a maximum matching in the conflict subgraph to get vertex cover lower bound
    #
    # For 256 vertices, let me just use the trivial upper bound and rely on good ordering.
    # With bitsets, the search should be fast enough.
    
    import sys
    sys.setrecursionlimit(10000)
    
    best_size = [0]
    best_mask = [0]
    
    # Precompute: for each vertex, its conflict bitset
    # Already have conflict[i]
    
    def popcount(x):
        return bin(x).count('1')
    
    def solve(candidates, current_size, current_mask):
        if time.time() > DEADLINE:
            return
        
        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            return
        
        # Trivial upper bound
        if current_size + popcount(candidates) <= best_size[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with highest degree in the candidate subgraph
        # For simplicity, pick the lowest set bit
        # Better: pick the vertex that appears in the most conflicts among candidates
        
        # Find the first set bit
        v = (candidates & -candidates).bit_length() - 1
        
        # Option 1: include v
        new_candidates = candidates & ~(1 << v) & ~conflict[v]
        solve(new_candidates, current_size + 1, current_mask | (1 << v))
        
        # Check deadline
        if time.time() > DEADLINE:
            return
        
        # Option 2: exclude v
        new_candidates2 = candidates & ~(1 << v)
        solve(new_candidates2, current_size, current_mask)
    
    # Initial candidates: all 256 vertices
    all_candidates = (1 << N) - 1
    solve(all_candidates, 0, 0)
    
    # Extract the result
    result = []
    mask = best_mask[0]
    for i in range(N):
        if mask & (1 << i):
            result.append(i + 1)
    
    if not result:
        # Fallback: just return odds
        result = list(range(1, 257, 2))
    
    return sorted(result)
