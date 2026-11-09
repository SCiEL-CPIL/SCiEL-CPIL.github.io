import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 16
    
    # Generate all possible edges in K16
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
    
    M = len(all_edges)  # 120 edges
    
    # Build conflict graph: two edges conflict if they share a vertex or form a triangle
    # Edge i = (u, v), edge j = (w, x)
    # Conflict if: u==w, u==x, v==w, v==x (share vertex) OR 
    #              (u,w,v) is a triangle, i.e., v==w or v==x or u==x or u==w (already covered)
    # Actually: edges (u,v) and (w,x) form a triangle if {u,v,w,x} has 3 vertices and
    #           the three edges exist. But since we're building the conflict graph for
    #           independent set, two edges conflict if they cannot both be in a triangle-free graph.
    #           They conflict if they share a vertex OR if adding both creates a triangle.
    #           Adding (u,v) and (w,x) creates a triangle if there's a third edge connecting
    #           two of the endpoints. But for independent set in conflict graph, we just need
    #           pairwise conflicts: (u,v) and (w,x) conflict if they share a vertex.
    #           Wait, that's not enough. Two edges that don't share a vertex can still form
    #           a triangle with a third edge. But for the independent set formulation, we need
    #           that no three edges form a triangle. This is a 3-wise constraint, not pairwise.
    #           
    # Let me reconsider. The problem is: find the maximum set of edges such that no three
    # form a triangle. This is equivalent to finding a maximum independent set in a 3-uniform
    # hypergraph, which is harder.
    #
    # Alternative: Use branch and bound directly on edges with triangle checking.
    
    # Precompute for each pair of edges whether they can coexist (don't form a triangle with any edge)
    # Actually, let's use a different approach: branch and bound with bitmasks
    
    # For each edge (u,v), precompute which other edges are "compatible" 
    # Two edges (u,v) and (w,x) are compatible if they don't share a vertex
    # AND there's no vertex z such that (u,z) and (v,z) are both edges (which would form triangle with (u,v))
    # Wait, that's still not right. Let me think again.
    
    # The constraint is: no three edges form a triangle. So we need to check triples.
    # This is a 3-uniform constraint, making it a 3-dimensional matching problem variant.
    
    # Let me use a simpler branch and bound approach:
    # - Maintain current edge set
    # - For each candidate edge, check if adding it creates a triangle
    # - Use upper bound: current_count + remaining_candidates_that_are_compatible
    
    # Precompute adjacency for fast triangle checking
    # For each pair of vertices (u,v), precompute the set of common neighbors
    # If we add edge (u,v), it creates a triangle if there exists w such that (u,w) and (v,w) are both in the set
    
    # Let's use bitmasks for efficiency
    # adj_mask[u] = bitmask of vertices adjacent to u in current graph
    
    # For branch and bound, we need a good upper bound.
    # Upper bound = current_edge_count + max_additional_edges
    # max_additional_edges can be bounded by: for each remaining edge, check if it's compatible
    # with all current edges. But this is still O(M^2) per node.
    
    # Better approach: Use the fact that in a triangle-free graph on n vertices,
    # the maximum number of edges is floor(n^2/4) = 64 for n=16.
    # So we can use 64 as an upper bound, but that's weak.
    
    # Let me try a different strategy: Start from a known 4-chromatic triangle-free graph
    # (like Grötzsch) and try to add edges greedily with backtracking.
    
    # Grötzsch graph (Mycielski of C5)
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # C5
        (5, 0), (5, 2),
        (6, 1), (6, 3),
        (7, 2), (7, 4),
        (8, 3), (8, 0),
        (9, 4), (9, 1),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # Initialize adjacency
    adj = [0] * N
    for u, v in groetzsch_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    base_edge_count = len(groetzsch_edges)  # 20 edges
    
    # Find all edges not in the base graph
    candidate_edges = []
    for u, v in all_edges:
        if not (adj[u] & (1 << v)):
            candidate_edges.append((u, v))
    
    best_adj = adj[:]
    best_edge_count = base_edge_count
    
    # Sort candidates by degree (higher degree first might be better for pruning)
    # Actually, let's try a different order: edges that are "more constrained" first
    
    def is_triangle_free(adj, u, v):
        """Check if adding edge (u,v) creates a triangle"""
        return (adj[u] & adj[v]) == 0
    
    def count_edges(adj):
        count = 0
        for i in range(N):
            count += bin(adj[i]).count('1')
        return count // 2
    
    # Branch and bound with time limit
    def branch_and_bound(adj, candidates, current_count, depth):
        nonlocal best_adj, best_edge_count, deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_count + len(candidates)
        # This is weak, but let's use it
        if current_count + len(candidates) <= best_edge_count:
            return
        
        if not candidates:
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_adj = adj[:]
            return
        
        # Choose the next edge to consider
        # Heuristic: choose edge with fewest conflicts (most compatible with others)
        # For simplicity, let's just take the first one, but we could optimize this
        
        # Try adding the edge
        u, v = candidates[0]
        remaining = candidates[1:]
        
        if is_triangle_free(adj, u, v):
            # Add the edge
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            # Filter remaining candidates: remove edges that would create triangles
            # with the new edge or are incompatible
            filtered = []
            for w, x in remaining:
                # Check if (w,x) is compatible with (u,v)
                # They conflict if they share a vertex or form a triangle
                if w == u or w == v or x == u or x == v:
                    continue
                # Check if adding both creates a triangle
                # Triangle if there's a common neighbor of u and v that's also neighbor of w or x
                # Actually, we need to check if (u,v) and (w,x) can coexist without forming a triangle
                # They form a triangle if there's a vertex z such that (u,z), (v,z), (w,z) or similar
                # This is complex. Let's just check if (w,x) is triangle-free given new_adj
                if is_triangle_free(new_adj, w, x):
                    filtered.append((w, x))
            
            branch_and_bound(new_adj, filtered, current_count + 1, depth + 1)
        
        # Try not adding the edge
        branch_and_bound(adj, remaining, current_count, depth + 1)
    
    # Start the search
    branch_and_bound(adj, candidate_edges, base_edge_count, 0)
    
    # Convert best_adj to list of edges
    result_edges = []
    for u in range(N):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                result_edges.append([u, v])
            mask ^= lsb
    
    return result_edges
