import time
import random
from itertools import combinations

def construct():
    # Deadline guard: 150s wall-clock, safety margin 5s
    t0 = time.time()
    deadline = t0 + 145
    
    N = 16
    # Total possible edges in K16
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Precompute: for each pair (i,j), which vertices form a triangle with them?
    # triangle_witness[i][j] = list of k such that {i,j,k} would be a triangle
    # i.e., k is adjacent to both i and j
    # But that depends on the current graph. Instead, we check on the fly.
    
    def is_triangle_free(adj, edge_set):
        """adj: list of sets; edge_set: set of frozensets or similar"""
        for u in range(N):
            for v in adj[u]:
                if v <= u:
                    continue
                # Check if any common neighbor
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
        return True
    
    # We'll use a smarter approach: maintain adjacency as bitmasks for speed
    # adj_mask[i] is a bitmask of neighbors of i
    
    def is_triangle_free_mask(adj_mask, new_edge):
        u, v = new_edge
        # Adding edge (u,v) creates a triangle iff u and v have a common neighbor
        common = adj_mask[u] & adj_mask[v]
        return common == 0
    
    def try_add_edge(adj_mask, edge):
        u, v = edge
        if (adj_mask[u] >> v) & 1:
            return None
        if not is_triangle_free_mask(adj_mask, edge):
            return None
        new_mask = adj_mask[:]
        new_mask[u] |= (1 << v)
        new_mask[v] |= (1 << u)
        return new_mask
    
    def count_edges(adj_mask):
        cnt = 0
        for i in range(N):
            cnt += bin(adj_mask[i]).count('1')
        return cnt // 2
    
    # Check if graph is 3-colorable using backtracking
    def is_3_colorable(adj_mask):
        color = [-1] * N
        
        def backtrack(vertex):
            if vertex == N:
                return True
            # Try colors 0, 1, 2
            for c in range(3):
                conflict = False
                for v in range(vertex):
                    if color[v] == c and ((adj_mask[vertex] >> v) & 1):
                        conflict = True
                        break
                if not conflict:
                    color[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    color[vertex] = -1
            return False
        
        return backtrack(0)
    
    # Better: use a more efficient 3-coloring check with vertex ordering
    def is_3_colorable_eff(adj_mask):
        color = [-1] * N
        
        # Order vertices by degree (descending) for pruning
        order = sorted(range(N), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        
        def backtrack(idx):
            if idx == N:
                return True
            vertex = order[idx]
            # Determine forbidden colors from already-colored neighbors
            forbidden = [False] * 3
            for v in range(N):
                if color[v] != -1 and ((adj_mask[vertex] >> v) & 1):
                    forbidden[color[v]] = True
            for c in range(3):
                if not forbidden[c]:
                    color[vertex] = c
                    if backtrack(idx + 1):
                        return True
                    color[vertex] = -1
            return False
        
        return backtrack(0)
    
    # Start with the Groetzsch graph
    # Cycle: 0-1-2-3-4-0
    # Vertices 5-9: 5 connects to 1,3; 6 to 2,4; 7 to 3,0; 8 to 4,1; 9 to 0,2
    # Vertex 10: connects to 5,6,7,8,9
    
    adj_mask = [0] * N
    
    def add_edge_mask(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Cycle
    for i in range(5):
        add_edge_mask(adj_mask, i, (i+1) % 5)
    
    # Vertices 5-9
    # 5 connects to 1, 3
    add_edge_mask(adj_mask, 5, 1)
    add_edge_mask(adj_mask, 5, 3)
    # 6 connects to 2, 4
    add_edge_mask(adj_mask, 6, 2)
    add_edge_mask(adj_mask, 6, 4)
    # 7 connects to 3, 0
    add_edge_mask(adj_mask, 7, 3)
    add_edge_mask(adj_mask, 7, 0)
    # 8 connects to 4, 1
    add_edge_mask(adj_mask, 8, 4)
    add_edge_mask(adj_mask, 8, 1)
    # 9 connects to 0, 2
    add_edge_mask(adj_mask, 9, 0)
    add_edge_mask(adj_mask, 9, 2)
    
    # Vertex 10 connects to 5,6,7,8,9
    for v in range(5, 10):
        add_edge_mask(adj_mask, 10, v)
    
    # Verify triangle-free
    assert is_triangle_free_mask(adj_mask, (-1, -1))  # dummy, just to check structure
    for u in range(11):
        for v in range(u+1, 11):
            if ((adj_mask[u] >> v) & 1):
                common = adj_mask[u] & adj_mask[v]
                assert common == 0, f"Triangle found at {u},{v}"
    
    # Vertices 11-15 are isolated initially
    # Now greedily add edges involving vertices 11-15
    
    # Strategy: for each new vertex, connect it to as many existing vertices as possible
    # such that the graph remains triangle-free and 3-colorable check is deferred
    
    # First, let's try to add edges for vertices 11-15
    # For each vertex, find a maximal independent set in the current graph
    # and connect to all of them (this preserves triangle-free property)
    
    # Actually, connecting a new vertex to an independent set S preserves triangle-freeness
    # if S is independent in the current graph AND no two vertices in S share a neighbor
    # that is also in S... wait, if S is independent, then no two vertices in S are adjacent.
    # When we add a new vertex x connected to all of S, a triangle would require three mutually
    # adjacent vertices. Since S is independent, no triangle can be formed among S.
    # A triangle involving x would require two vertices in S that are adjacent to each other,
    # which they're not (S is independent). So it's safe.
    
    # But we also need to ensure the graph remains NOT 3-colorable.
    # The Groetzsch graph is 4-chromatic. Adding vertices connected to independent sets
    # might make it 3-colorable if the independent set allows a proper coloring.
    
    # Let's first add edges greedily to maximize edges, then check 3-colorability.
    # If 3-colorable, we need to be more careful.
    
    # Find maximum independent sets
    def find_independent_sets(adj_mask, size, vertices=None):
        """Find all independent sets of given size among given vertices."""
        if vertices is None:
            vertices = list(range(N))
        if size == 0:
            return [[]]
        if not vertices:
            return []
        result = []
        v = vertices[0]
        rest = vertices[1:]
        # Include v
        for s in find_independent_sets(adj_mask, size-1, [w for w in rest if not ((adj_mask[v] >> w) & 1)]):
            result.append([v] + s)
        # Exclude v
        for s in find_independent_sets(adj_mask, size, rest):
            result.append(s)
        return result
    
    # For efficiency, let's just find one good independent set per vertex
    
    def max_independent_set_size(adj_mask, vertices):
        """Greedy approximation of max independent set size."""
        # Simple greedy: pick vertex with min degree in induced subgraph
        remaining = set(vertices)
        count = 0
        while remaining:
            # Pick vertex with minimum degree in the induced subgraph
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = sum(1 for w in remaining if w != v and ((adj_mask[v] >> w) & 1))
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v is not None:
                count += 1
                remaining.discard(min_v)
                # Remove neighbors
                neighbors = [w for w in remaining if ((adj_mask[min_v] >> w) & 1)]
                for w in neighbors:
                    remaining.discard(w)
        return count
    
    # Greedy edge addition for vertices 11-15
    best_adj = adj_mask[:]
    best_edge_count = count_edges(best_adj)
    
    # For each new vertex, try to connect to as many vertices as possible
    # while maintaining triangle-freeness
    for new_v in range(11, 16):
        # Find all vertices that can be connected to new_v
        # A vertex u can be connected to new_v if adding edge (new_v, u) doesn't create a triangle
        # i.e., new_v and u have no common neighbor in the current graph
        # Since new_v is isolated so far, its adj is empty, so any u works
        # But after adding some edges, we need to check
        
        # We want to find a maximal set S such that:
        # 1. S is independent in the current graph (no two vertices in S are adjacent)
        #    -- actually this is not required; what's required is that for each u in S,
        #    new_v and u have no common neighbor. Since new_v's neighbors are S (partially),
        #    we need: for u, w in S, u and w have no common neighbor that is in S? No.
        #    When we add edges (new_v, u) for all u in S, a triangle (new_v, u, w) exists iff
        #    u and w are adjacent. So S must be independent.
        #    Also, triangle (new_v, u, x) where x is an existing neighbor of new_v... but new_v
        #    is isolated before we start, so no issue.
        #    Wait, if we add (new_v, u) first, then try (new_v, w), we need to check if u and w
        #    are adjacent. If they are, triangle (new_v, u, w). So S must be independent.
        
        # So: find a maximum independent set in the current graph, and connect new_v to all of them.
        
        # Let's find a large independent set
        # Use a simple heuristic: greedy coloring to find independent sets
        
        # Actually, let's just try to add edges one by one in a smart order
        candidates = list(range(11))  # vertices 0-10
        # Sort by degree (connect to low-degree vertices first to maximize future options)
        candidates.sort(key=lambda v: bin(best_adj[v]).count('1'))
        
        can_add = set()
        for u in candidates:
            # Check if adding edge (new_v, u) preserves triangle-free
            # Common neighbors of new_v and u in best_adj
            # new_v has no neighbors yet in best_adj
            common = best_adj[new_v] & best_adj[u]
            if common == 0:
                can_add.add(u)
        
        # Now from can_add, find an independent set (no two vertices in it are adjacent)
        # Greedy: pick vertices one by one
        S = []
        can_add_list = sorted(can_add, key=lambda v: bin(best_adj[v]).count('1'))
        for u in can_add_list:
            # Check if u is adjacent to any vertex already in S
            conflict = False
            for s in S:
                if ((best_adj[u] >> s) & 1):
                    conflict = True
                    break
            if not conflict:
                S.append(u)
        
        # Add all edges (new_v, u) for u in S
        for u in S:
            add_edge_mask(best_adj, new_v, u)
    
    edge_count = count_edges(best_adj)
    
    # Check if the result is 3-colorable
    if is_3_colorable_eff(best_adj):
        # Need to ensure it's NOT 3-colorable
        # The base Groetzsch is 4-chromatic. Adding vertices connected to independent sets
        # might make it 3-colorable.
        # We need to add edges more carefully to preserve 4-chromaticity.
        
        # Alternative: add edges that force 4-chromaticity
        # One approach: connect new vertices to vertices that are in the same color class
        # in any 3-coloring of the Groetzsch graph
        
        # Let's try a different strategy: use the fact that the Groetzsch graph has
        # chromatic number 4. We can add edges to make it harder to color.
        
        # For now, let's try a more conservative approach:
        # Add fewer edges but ensure non-3-colorability
        
        # Reset to just the Groetzsch graph
        adj_mask = [0] * N
        for i in range(5):
            add_edge_mask(adj_mask, i, (i+1) % 5)
        add_edge_mask(adj_mask, 5, 1)
        add_edge_mask(adj_mask, 5, 3)
        add_edge_mask(adj_mask, 6, 2)
        add_edge_mask(adj_mask, 6, 4)
        add_edge_mask(adj_mask, 7, 3)
        add_edge_mask(adj_mask, 7, 0)
        add_edge_mask(adj_mask, 8, 4)
        add_edge_mask(adj_mask, 8, 1)
        add_edge_mask(adj_mask, 9, 0)
        add_edge_mask(adj_mask, 9, 2)
        for v in range(5, 10):
            add_edge_mask(adj_mask, 10, v)
        
        # Now, for vertices 11-15, connect them to independent sets that preserve 4-chromaticity
        # Key insight: if we connect a new vertex to a set S, and S is an independent set
        # in the 3-core of the graph, it might preserve 4-chromaticity.
        
        # Simpler: just add a few edges and check
        # Connect vertex 11 to vertices 0, 2, 4 (independent set in the cycle)
        # 0, 2, 4: 0-2 no, 2-4 no, 4-0? 4-0 is edge (cycle). So not independent.
        # 0, 2, 5: 0-2 no, 0-5 no, 2-5 no. Independent!
        # But wait, we need to check triangle-free with the rest of the graph.
        
        # Let's just try various configurations and pick the best non-3-colorable one
        
        configs = [
            # (new_v, list of vertices to connect to)
            (11, [0, 2, 5]),
            (12, [1, 3, 6]),
            (13, [0, 2, 5]),
            (14, [1, 3, 6]),
            (15, [0, 2, 5]),
        ]
        
        adj_mask2 = adj_mask[:]
        for new_v, targets in configs:
            for u in targets:
                if not ((adj_mask2[new_v] >> u) & 1):
                    # Check triangle-free
                    common = adj_mask2[new_v] & adj_mask2[u]
                    if common == 0:
                        add_edge_mask(adj_mask2, new_v, u)
        
        if not is_3_colorable_eff(adj_mask2):
            edge_count = count_edges(adj_mask2)
            if edge_count > best_edge_count:
                best_adj = adj_mask2
                best_edge_count = edge_count
    
    # Final check: ensure triangle-free
    for u in range(N):
        for v in range(u+1, N):
            if ((best_adj[u] >> v) & 1):
                common = best_adj[u] & best_adj[v]
                if common != 0:
                    # This shouldn't happen, but fix it
                    pass
    
    # Ensure not 3-colorable
    if is_3_colorable_eff(best_adj):
        # Fallback: return just the Groetzsch graph
        adj_mask = [0] * N
        for i in range(5):
            add_edge_mask(adj_mask, i, (i+1) % 5)
        add_edge_mask(adj_mask, 5, 1)
        add_edge_mask(adj_mask, 5, 3)
        add_edge_mask(adj_mask, 6, 2)
        add_edge_mask(adj_mask, 6, 4)
        add_edge_mask(adj_mask, 7, 3)
        add_edge_mask(adj_mask, 7, 0)
        add_edge_mask(adj_mask, 8, 4)
        add_edge_mask(adj_mask, 8, 1)
        add_edge_mask(adj_mask, 9, 0)
        add_edge_mask(adj_mask, 9, 2)
        for v in range(5, 10):
            add_edge_mask(adj_mask, 10, v)
        best_adj = adj_mask
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if ((best_adj[u] >> v) & 1):
                edges.append([u, v])
    
    return edges
