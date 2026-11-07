def sign_edges(graph, budget):
    import numpy as np
    import time
    import itertools
    
    start_time = time.time()
    DEADLINE = 85  # Safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Precompute edge endpoints as arrays for fast access
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Identify edge types: top cycle, bottom cycle, matching
    # Prism graph: vertices 0..m-1 (top), m..2m-1 (bottom)
    # Edges: (i, i+1) for i in 0..m-2, (m-1, 0) for top cycle
    #         (m+i, m+i+1) for i in 0..m-2, (2m-1, m) for bottom cycle
    #         (i, m+i) for i in 0..m-1 for matching
    edge_type = np.zeros(n_edges, dtype=np.int32)  # 0=top, 1=bottom, 2=matching
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            edge_type[i] = 0  # top cycle
        elif u >= m and v >= m:
            edge_type[i] = 1  # bottom cycle
        else:
            edge_type[i] = 2  # matching
    
    # Build base adjacency matrix structure
    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius_from_A(A):
        return np.max(np.abs(np.linalg.eigvalsh(A)))
    
    # Fast validity check using the fact that for prism graphs,
    # the lift is connected iff the product of signs around the "prism cycle" is -1.
    # The prism cycle goes: top cycle (m edges) + one matching edge + bottom cycle reversed (m edges) + one matching edge
    # More simply: the lift is disconnected iff there exists a vertex labeling s:V->{-1,1}
    # such that sign(u,v) = s[u]*s[v] for all edges. This is equivalent to the signing being "balanced".
    # For efficiency, use BFS with early termination.
    
    def is_valid_lift_fast(signs):
        # BFS on the 2-lift
        # Vertex (v, side) -> (v', side XOR (sign==-1))
        adj = [[] for _ in range(2 * n_vertices)]
        for i in range(n_edges):
            u, v = edge_u[i], edge_v[i]
            s = signs[i]
            if s == 1:
                adj[2*u].append(2*v)
                adj[2*v].append(2*u)
                adj[2*u+1].append(2*v+1)
                adj[2*v+1].append(2*u+1)
            else:
                adj[2*u].append(2*v+1)
                adj[2*v+1].append(2*u)
                adj[2*u+1].append(2*v)
                adj[2*v].append(2*u+1)
        
        visited = np.zeros(2 * n_vertices, dtype=bool)
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    if count == 2 * n_vertices:
                        return True
                    stack.append(nb)
        return count == 2 * n_vertices
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    def evaluate_and_update(signs):
        nonlocal best_signs, best_radius
        if not is_valid_lift_fast(signs):
            return
        A = build_signed_adj(signs)
        radius = spectral_radius_from_A(A)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Phase 1: Structured pattern search
    # For prism graphs, we can parameterize signs by patterns on the three edge types.
    # Try all combinations of constant signs on each edge type, plus some variations.
    
    # First, try all 8 combinations of constant signs on top/bottom/matching
    for top_s in [-1, 1]:
        for bot_s in [-1, 1]:
            for match_s in [-1, 1]:
                signs = np.ones(n_edges, dtype=int)
                for i in range(n_edges):
                    if edge_type[i] == 0:
                        signs[i] = top_s
                    elif edge_type[i] == 1:
                        signs[i] = bot_s
                    else:
                        signs[i] = match_s
                evaluate_and_update(signs)
    
    # Phase 2: Try patterns with one edge flipped per type
    # For each edge type, try flipping one edge and see if it improves
    base_patterns = []
    for top_s in [-1, 1]:
        for bot_s in [-1, 1]:
            for match_s in [-1, 1]:
                base_patterns.append((top_s, bot_s, match_s))
    
    for top_s, bot_s, match_s in base_patterns:
        # Try flipping individual edges of each type
        for i in range(n_edges):
            if time.time() - start_time > DEADLINE - 10:
                break
            signs = np.ones(n_edges, dtype=int)
            for j in range(n_edges):
                if edge_type[j] == 0:
                    signs[j] = top_s
                elif edge_type[j] == 1:
                    signs[j] = bot_s
                else:
                    signs[j] = match_s
            signs[i] *= -1  # Flip this edge
            evaluate_and_update(signs)
    
    # Phase 3: Local search from best found so far
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_A = build_signed_adj(current_signs)
        current_radius = best_radius
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            # Try 1-flip
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 1:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                
                # Update A in-place
                u, v = edge_u[i], edge_v[i]
                current_A[u, v] = current_signs[i]
                current_A[v, u] = current_signs[i]
                
                new_radius = spectral_radius_from_A(current_A)
                
                if new_radius < current_radius - 1e-9 and is_valid_lift_fast(current_signs):
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old
                    current_A[u, v] = old
                    current_A[v, u] = old
                    
            # If no 1-flip improved, try 2-flips (stochastic)
            if not improved:
                for _ in range(30):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    oi = current_signs[i]
                    oj = current_signs[j]
                    current_signs[i] = -oi
                    current_signs[j] = -oj
                    
                    ui, vi = edge_u[i], edge_v[i]
                    uj, vj = edge_u[j], edge_v[j]
                    current_A[ui, vi] = current_signs[i]
                    current_A[vi, ui] = current_signs[i]
                    current_A[uj, vj] = current_signs[j]
                    current_A[vj, uj] = current_signs[j]
                    
                    new_radius = spectral_radius_from_A(current_A)
                    
                    if new_radius < current_radius - 1e-9 and is_valid_lift_fast(current_signs):
                        current_radius = new_radius
                        improved = True
                    else:
                        current_signs[i] = oi
                        current_signs[j] = oj
                        current_A[ui, vi] = oi
                        current_A[vi, ui] = oi
                        current_A[uj, vj] = oj
                        current_A[vj, uj] = oj
                        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
    
    # Phase 4: More random restarts with local search
    for restart in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        if best_signs is not None:
            signs = best_signs.copy()
            n_flips = rng.integers(1, 5)
            idxs = rng.choice(n_edges, size=n_flips, replace=False)
            signs[idxs] *= -1
        else:
            signs = rng.choice([-1, 1], size=n_edges)
        
        if not is_valid_lift_fast(signs):
            continue
            
        current_signs = signs.copy()
        current_A = build_signed_adj(current_signs)
        current_radius = spectral_radius_from_A(current_A)
        
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 1:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                
                u, v = edge_u[i], edge_v[i]
                current_A[u, v] = current_signs[i]
                current_A[v, u] = current_signs[i]
                
                new_radius = spectral_radius_from_A(current_A)
                
                if new_radius < current_radius - 1e-9 and is_valid_lift_fast(current_signs):
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old
                    current_A[u, v] = old
                    current_A[v, u] = old
                    
            if not improved:
                for _ in range(20):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    oi = current_signs[i]
                    oj = current_signs[j]
                    current_signs[i] = -oi
                    current_signs[j] = -oj
                    
                    ui, vi = edge_u[i], edge_v[i]
                    uj, vj = edge_u[j], edge_v[j]
                    current_A[ui, vi] = current_signs[i]
                    current_A[vi, ui] = current_signs[i]
                    current_A[uj, vj] = current_signs[j]
                    current_A[vj, uj] = current_signs[j]
                    
                    new_radius = spectral_radius_from_A(current_A)
                    
                    if new_radius < current_radius - 1e-9 and is_valid_lift_fast(current_signs):
                        current_radius = new_radius
                        improved = True
                    else:
                        current_signs[i] = oi
                        current_signs[j] = oj
                        current_A[ui, vi] = oi
                        current_A[vi, ui] = oi
                        current_A[uj, vj] = oj
                        current_A[vj, uj] = oj
                        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    if best_signs is None:
        # Fallback: find a valid configuration
        best_signs = rng.choice([-1, 1], size=n_edges)
        for _ in range(1000):
            if is_valid_lift_fast(best_signs):
                break
            idx = rng.integers(0, n_edges)
            best_signs[idx] *= -1

    return best_signs
