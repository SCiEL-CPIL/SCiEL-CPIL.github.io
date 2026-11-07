def sign_edges(graph, budget):
    import numpy as np
    
    n = graph['n']
    edges = list(graph['edges'])
    E = len(edges)
    m = n // 2
    
    # Build adjacency list with edge indices
    adj = {v: [] for v in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Find spanning tree via BFS
    from collections import deque
    visited = {0}
    tree_edge_set = set()
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for (v, idx) in adj[u]:
            if v not in visited:
                visited.add(v)
                tree_edge_set.add(idx)
                queue.append(v)
    # The graph is connected (prism), so visited = all vertices
    # Chord edges = all edges not in tree
    chord_indices = [idx for idx in range(E) if idx not in tree_edge_set]
    num_chords = len(chord_indices)
    
    # Precompute: for each chord index, its position in chord_indices
    # We'll enumerate 2^num_chords assignments
    
    # For BFS balance check, we need adjacency with edge indices (already have)
    
    best_radius = float('inf')
    best_signs = None
    
    # Precompute eigvalsh cost: n x n matrix
    # For each assignment:
    for mask in range(1 << num_chords):
        # Build sign array
        signs = np.ones(E, dtype=np.int8)
        for j, cidx in enumerate(chord_indices):
            if (mask >> j) & 1:
                signs[cidx] = -1
        # Tree edges remain +1
        
        # Check balanced via BFS
        tau = np.zeros(n, dtype=np.int8)  # 0 = unvisited, else +1/-1
        tau[0] = 1
        balanced = True
        q = deque([0])
        while q and balanced:
            u = q.popleft()
            for (v, idx) in adj[u]:
                expected = tau[u] * signs[idx]
                if tau[v] == 0:
                    tau[v] = expected
                    q.append(v)
                else:
                    if tau[v] != expected:
                        balanced = False
                        break
        if balanced:
            continue  # skip balanced (lift would be disconnected)
        
        # Build signed adjacency
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        vals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(vals))
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all -1 (should be unbalanced for prism? check)
        # Actually need a valid unbalanced assignment. Use a single -1 edge.
        best_signs = np.ones(E, dtype=np.int8)
        best_signs[0] = -1
        # verify unbalanced - a single -1 on an edge in a cycle? 
        # For safety, do a proper unbalanced: set all +1 except make one cycle unbalanced
        # Simpler: set signs[0]=-1 and ensure it's in a cycle. Edge 0 is a cycle edge, so the cycle containing it has product -1 -> unbalanced. Good.
    
    return best_signs.astype(np.int8)
