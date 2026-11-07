def sign_edges(graph, budget):
    import numpy as np
    import time
    from itertools import product

    deadline = time.time() + 75.0

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Build edge index
    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i

    def key(u, v):
        return (min(u, v), max(u, v))

    # Identify edge categories
    # Bottom cycle: (i, i+1) for i in 0..m-1
    # Top cycle: (m+i, m+i+1) for i in 0..m-1
    # Matching: (i, m+i) for i in 0..m-1
    
    bottom_edges = []  # indices of bottom cycle edges
    top_edges = []     # indices of top cycle edges
    match_edges = []   # indices of matching edges

    for i in range(m):
        bottom_edges.append(idx[key(i, (i+1)%m)])
        top_edges.append(idx[key(m+i, m+(i+1)%m)])
        match_edges.append(idx[key(i, m+i)])

    # Gauge invariance: fix a spanning tree to +1
    # A spanning tree for prism graph:
    # - Bottom cycle: edges 0 to m-2 (all except the last one closing the cycle)
    # - Top cycle: edges 0 to m-2 (all except the last one)
    # - Matching: all m edges
    # 
    # Actually, let's think more carefully. The prism graph has 2m vertices and 3m edges.
    # A spanning tree has 2m-1 edges. So we fix 2m-1 edges to +1, leaving m+1 edges free.
    #
    # Let's choose the spanning tree as:
    # - All bottom cycle edges except the last: m-1 edges
    # - All top cycle edges except the last: m-1 edges  
    # - All matching edges: m edges
    # Total: (m-1) + (m-1) + m = 3m - 2 edges
    # But we need 2m-1 edges for a spanning tree. That's too many.
    #
    # Let me reconsider. The prism graph:
    # Vertices: 0..m-1 (bottom), m..2m-1 (top)
    # Edges: bottom cycle (m edges), top cycle (m edges), matching (m edges)
    # Total edges: 3m, vertices: 2m
    # Spanning tree: 2m-1 edges
    #
    # A valid spanning tree:
    # - Bottom path: edges (0,1), (1,2), ..., (m-2, m-1): m-1 edges
    # - Top path: edges (m, m+1), ..., (2m-2, 2m-1): m-1 edges
    # - Matching: all m edges
    # Total: (m-1) + (m-1) + m = 3m - 2
    # Still too many. We need to remove m-1 more edges.
    #
    # Actually, if we take all matching edges and the two paths, we have a connected graph
    # with 2m vertices and 3m-2 edges. This is not a tree; it has cycles.
    #
    # Let me think differently. The cycle space has dimension E - V + 1 = 3m - 2m + 1 = m+1.
    # So there are m+1 independent cycles, and correspondingly m+1 "chord" edges that can be
    # used to parameterize the sign assignments up to gauge.
    #
    # Gauge fixing: pick a spanning tree T with 2m-1 edges, fix their signs to +1.
    # The remaining E - (2m-1) = 3m - 2m + 1 = m+1 edges are the "chords".
    #
    # Let's construct a specific spanning tree:
    # - All m matching edges: (i, m+i) for i=0..m-1
    # - Bottom path: (0,1), (1,2), ..., (m-2, m-1): m-1 edges
    # - Top: just one edge to connect top to bottom via matching, but matching already does that.
    #   Actually, with all matching edges and bottom path, we have:
    #   - Bottom vertices 0..m-1 connected by path
    #   - Each bottom vertex i connected to top vertex m+i
    #   - Top vertices m..2m-1 are each connected to their bottom counterpart
    #   This gives a connected graph on 2m vertices with m + (m-1) = 2m-1 edges. 
    #   Yes! This is a spanning tree.
    #
    # So the spanning tree T consists of:
    # - Matching edges: all m
    # - Bottom cycle edges: first m-1 (indices 0 to m-2 in bottom_edges)
    #
    # The chords (non-tree edges) are:
    # - Bottom cycle edge m-1 (the last one closing the cycle)
    # - All top cycle edges: m edges
    # Total chords: 1 + m = m+1

    tree_edges = set()
    for ei in match_edges:
        tree_edges.add(ei)
    for i in range(m - 1):
        tree_edges.add(bottom_edges[i])

    chord_edges = []
    for i in range(E):
        if i not in tree_edges:
            chord_edges.append(i)

    # There should be m+1 chord edges
    # assert len(chord_edges) == m + 1

    def is_balanced_fast(sgn):
        # Check that all fundamental cycles have product +1
        # For a 2-lift to be valid, the product of signs around every cycle must be +1
        # This is equivalent to: the sign assignment is a "coboundary" or the lift is connected
        # 
        # Actually, for the lift to be connected, we need that the product of signs
        # around every cycle is +1. Equivalently, there exists a vertex labeling 
        # s: V -> {+1, -1} such that for every edge (u,v), sign(u,v) = s(u)*s(v).
        #
        # Let's check this by BFS from vertex 0.
        # Assign s[0] = 1. Then for each edge (u,v), s(v) = sign(u,v) * s(u).
        # If we get a contradiction, it's not a valid coboundary, hence the lift is disconnected.
        
        s = np.zeros(n, dtype=np.int8)
        s[0] = 1
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        
        from collections import deque
        dq = deque([0])
        
        while dq:
            u = dq.popleft()
            for i, (a, b) in enumerate(edges):
                v = b if a == u else a
                if not visited[v]:
                    s[v] = sgn[i] * s[u]
                    visited[v] = True
                    dq.append(v)
                else:
                    # Check consistency
                    expected = sgn[i] * s[u]
                    if s[v] != expected:
                        return False
        
        return True

    def spectral_radius(sgn):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = sgn[i]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    # Exhaustive enumeration over 2^(m+1) chord sign assignments
    # For m <= 9, m+1 <= 10, so 2^10 = 1024 assignments. Very fast.
    
    best_signs = None
    best_radius = float('inf')

    num_chords = len(chord_edges)
    total_assignments = 2 ** num_chords

    for mask in range(total_assignments):
        if time.time() > deadline - 5:
            break
        
        sgn = np.ones(E, dtype=np.int8)
        for j, ei in enumerate(chord_edges):
            if mask & (1 << j):
                sgn[ei] = -1
        
        # Check validity
        if not is_balanced_fast(sgn):
            continue
        
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    if best_signs is None:
        # Fallback: try all +1, then flip one edge if needed
        best_signs = np.ones(E, dtype=np.int8)
        if not is_balanced_fast(best_signs):
            # Try flipping each chord edge
            for ei in chord_edges:
                sgn = np.ones(E, dtype=np.int8)
                sgn[ei] = -1
                if is_balanced_fast(sgn):
                    best_signs = sgn
                    break
            else:
                # Should not reach here for prism graphs
                best_signs[0] = -1

    return np.asarray(best_signs, dtype=np.int8)
