def sign_edges(graph, budget):
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Identify edge indices for bottom cycle, top cycle, and matching
    bot_edge_indices = []
    top_edge_indices = []
    match_edge_indices = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bot_edge_indices.append(i)
        elif u >= m and v >= m:
            top_edge_indices.append(i)
        else:
            match_edge_indices.append(i)
    
    def is_unbalanced(signs):
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
        pot = [0] * n
        pot[0] = 1
        stack = [0]
        while stack:
            u = stack.pop()
            pu = pot[u]
            for v, ei in adj[u]:
                s = signs[ei]
                want = pu * s
                if pot[v] == 0:
                    pot[v] = want
                    stack.append(v)
                elif pot[v] != want:
                    return True
        return False

    def spectral_radius(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    best_signs = None
    best_radius = float('inf')
    
    # Strategy: Enumerate all 2^m bottom patterns x 2^m top patterns, matching = +1
    # This gives 2^(2m) cases. For m=9, that's 2^18 = 262144, which is feasible.
    # For m=4,5,7: 2^8=256, 2^10=1024, 2^14=16384 -- all fast.
    
    for bot_mask in range(2 ** m):
        if time.time() > deadline:
            break
        bottom_signs = np.ones(m, dtype=np.int8)
        for j in range(m):
            if (bot_mask >> j) & 1:
                bottom_signs[j] = -1  # Use -1 for bit=1 to vary
        
        for top_mask in range(2 ** m):
            if time.time() > deadline:
                break
            top_signs = np.ones(m, dtype=np.int8)
            for j in range(m):
                if (top_mask >> j) & 1:
                    top_signs[j] = -1
            
            # Construct full sign vector
            signs = np.ones(E, dtype=np.int8)
            for k, ei in enumerate(bot_edge_indices):
                signs[ei] = bottom_signs[k]
            for k, ei in enumerate(top_edge_indices):
                signs[ei] = top_signs[k]
            # Matching signs are +1
            
            # Check validity (unbalanced)
            if not is_unbalanced(signs):
                continue
            
            # Calculate radius
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # Fallback if no valid sign found
    if best_signs is None:
        best_signs = np.ones(E, dtype=np.int8)
        best_signs[0] = -1
        if not is_unbalanced(best_signs):
            best_signs[1] = -1
            if not is_unbalanced(best_signs):
                best_signs[2] = -1

    return np.asarray(best_signs, dtype=np.int8)
