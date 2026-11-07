def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    start_time = time.time()
    DEADLINE = 78  # Safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Precompute adjacency: for each vertex, list of (edge_idx, other_vertex)
    adj = [[] for _ in range(2 * m)]
    for idx in range(n_edges):
        u, v = edge_u[idx], edge_v[idx]
        adj[u].append((idx, v))
        adj[v].append((idx, u))
    
    # Build spanning tree and fundamental cycles for fast balance checking
    # A signed graph is balanced iff every cycle has an even number of -1 edges
    # We use a spanning tree: for each non-tree edge, the fundamental cycle must have even # of -1s
    
    n_verts = 2 * m
    # Find spanning tree using BFS
    parent_edge = np.full(n_verts, -1, dtype=int)
    parent_vert = np.full(n_verts, -1, dtype=int)
    in_tree = [False] * n_verts
    in_tree[0] = True
    q = deque([0])
    tree_edges = set()
    while q:
        u = q.popleft()
        for eidx, v in adj[u]:
            if not in_tree[v]:
                in_tree[v] = True
                parent_edge[v] = eidx
                parent_vert[v] = u
                tree_edges.add(eidx)
                q.append(v)
    
    # Non-tree edges
    non_tree_edges = [i for i in range(n_edges) if i not in tree_edges]
    
    # For each non-tree edge, compute the fundamental cycle (list of edge indices)
    # and track parity (number of -1 edges mod 2)
    def get_fundamental_cycle(eidx):
        """Get the edges in the fundamental cycle for non-tree edge eidx."""
        u, v = edge_u[eidx], edge_v[eidx]
        # Trace from u to root
        cycle = [eidx]
        def trace_to_root(x):
            while parent_edge[x] != -1:
                cycle.append(parent_edge[x])
                x = parent_vert[x]
        trace_to_root(u)
        trace_to_root(v)
        return cycle
    
    # Precompute fundamental cycles
    fund_cycles = []
    for eidx in non_tree_edges:
        fund_cycles.append(get_fundamental_cycle(eidx))
    
    # For fast validity: lift is connected iff NOT balanced
    # Balanced iff for every fundamental cycle, the number of -1 edges is even
    # Precompute: for each edge, which fundamental cycles it appears in
    edge_in_cycles = [[] for _ in range(n_edges)]
    for ci, cyc in enumerate(fund_cycles):
        for eidx in cyc:
            edge_in_cycles[eidx].append(ci)
    
    def compute_cycle_parities(signs):
        """Return array of parities (0=even, 1=odd) for each fundamental cycle."""
        parities = np.zeros(len(fund_cycles), dtype=int)
        for ci, cyc in enumerate(fund_cycles):
            cnt = 0
            for eidx in cyc:
                if signs[eidx] == -1:
                    cnt += 1
            parities[ci] = cnt % 2
        return parities
    
    def is_valid_lift_fast(signs, parities):
        """Lift is connected iff not balanced, i.e., at least one cycle has odd # of -1s."""
        return np.any(parities == 1)
    
    def spectral_radius(signs):
        n = 2 * m
        A = np.zeros((n, n))
        for idx in range(n_edges):
            s = signs[idx]
            u, v = edge_u[idx], edge_v[idx]
            A[u, v] = s
            A[v, u] = s
        vals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(vals)))
    
    def local_search(initial_signs):
        signs = initial_signs.copy()
        parities = compute_cycle_parities(signs)
        if not is_valid_lift_fast(signs, parities):
            return None, float('inf')
        
        current_radius = spectral_radius(signs)
        
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 2:
            improved = False
            
            # Single flip moves with fast validity check
            for i in range(n_edges):
                if (time.time() - start_time) > DEADLINE - 2:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                
                # Update parities: flipping edge i changes parity of all cycles containing i
                new_parities = parities.copy()
                for ci in edge_in_cycles[i]:
                    new_parities[ci] ^= 1
                
                if is_valid_lift_fast(signs, new_parities):
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        parities = new_parities
                        improved = True
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign
            
            # Double flip moves (limited to keep runtime reasonable)
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Only try double flips on a subset to save time
            if n_edges <= 30:
                for i in range(n_edges):
                    if (time.time() - start_time) > DEADLINE - 2:
                        break
                    for j in range(i + 1, n_edges):
                        if (time.time() - start_time) > DEADLINE - 2:
                            break
                        si, sj = signs[i], signs[j]
                        signs[i] = -si
                        signs[j] = -sj
                        
                        new_parities = parities.copy()
                        for ci in edge_in_cycles[i]:
                            new_parities[ci] ^= 1
                        for ci in edge_in_cycles[j]:
                            new_parities[ci] ^= 1
                        
                        if is_valid_lift_fast(signs, new_parities):
                            new_radius = spectral_radius(signs)
                            if new_radius < current_radius - 1e-10:
                                current_radius = new_radius
                                parities = new_parities
                                improved = True
                            else:
                                signs[i] = si
                                signs[j] = sj
                        else:
                            signs[i] = si
                            signs[j] = sj
        
        return signs, current_radius
    
    # Build edge type mapping
    top_edges = []
    bot_edges = []
    match_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bot_edges.append(idx)
        else:
            match_edges.append(idx)
    
    rng = np.random.default_rng(42)
    best_signs = None
    best_radius = float('inf')
    
    for restart in range(200):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        
        if restart == 0:
            signs = np.ones(n_edges, dtype=int)
        elif restart == 1:
            signs = np.ones(n_edges, dtype=int)
            for i, eidx in enumerate(top_edges):
                if i % 2 == 1:
                    signs[eidx] = -1
        elif restart == 2:
            signs = np.ones(n_edges, dtype=int)
            for i, eidx in enumerate(bot_edges):
                if i % 2 == 1:
                    signs[eidx] = -1
        elif restart == 3:
            signs = np.ones(n_edges, dtype=int)
            for i, eidx in enumerate(match_edges):
                if i % 2 == 1:
                    signs[eidx] = -1
        elif restart == 4:
            signs = rng.choice([-1, 1], size=n_edges).astype(int)
        else:
            if best_signs is not None:
                signs = best_signs.copy()
                k = int(rng.integers(1, max(2, n_edges // 3)))
                flip_idx = rng.choice(n_edges, size=k, replace=False)
                signs[flip_idx] *= -1
            else:
                signs = rng.choice([-1, 1], size=n_edges).astype(int)
        
        s, r = local_search(signs)
        if s is not None and r < best_radius:
            best_signs = s
            best_radius = r
    
    if best_signs is None:
        # Fallback: find a valid configuration
        best_signs = np.ones(n_edges, dtype=int)
        if len(match_edges) > 0:
            best_signs[match_edges[0]] = -1
    
    return best_signs
