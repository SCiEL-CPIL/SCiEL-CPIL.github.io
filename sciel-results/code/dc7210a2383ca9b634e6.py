def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 85 # Safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Precompute edge indices for faster lookup
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
        
    def spectral_radius_from_A(A):
        return np.max(np.abs(np.linalg.eigvalsh(A)))
        
    def is_valid_lift(signs):
        # BFS on lift
        # Lift has 2*n_vertices nodes
        # Adjacency list
        adj = [[] for _ in range(2 * n_vertices)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            # u, v are base vertices
            # Lift vertices: 2*u, 2*u+1
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
                
        visited = [False] * (2 * n_vertices)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == 2 * n_vertices

    def make_valid_if_needed(signs):
        """Try to make signs valid by flipping edges if needed"""
        if is_valid_lift(signs):
            return signs
        # Try flipping random edges until valid
        for _ in range(100):
            idx = np.random.randint(n_edges)
            signs[idx] *= -1
            if is_valid_lift(signs):
                return signs
        # Fallback: use a known valid pattern
        # For prism graphs, having an odd number of -1 on the "vertical" matching edges
        # or on one of the cycles can help
        for _ in range(1000):
            signs = np.random.choice([-1, 1], size=n_edges)
            if is_valid_lift(signs):
                return signs
        return signs  # Should not reach here

    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    # Multiple restarts with different strategies
    for restart in range(200):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Initialize with various strategies
        if restart == 0:
            # All +1, then make valid
            signs = np.ones(n_edges, dtype=int)
        elif restart == 1:
            # Random
            signs = rng.choice([-1, 1], size=n_edges)
        elif restart == 2:
            # Pattern: alternate signs on one cycle
            signs = np.ones(n_edges, dtype=int)
            # Find edges in the first cycle (vertices 0..m-1)
            for i, (u, v) in enumerate(edges):
                if u < m and v < m:
                    signs[i] = 1 if (u % 2 == 0) else -1
        elif restart == 3:
            # Pattern: all -1 on vertical edges
            signs = np.ones(n_edges, dtype=int)
            for i, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    signs[i] = -1
        elif restart == 4:
            # Pattern: all -1 on one cycle
            signs = np.ones(n_edges, dtype=int)
            for i, (u, v) in enumerate(edges):
                if u < m and v < m:
                    signs[i] = -1
        else:
            # Perturb best found so far
            if best_signs is not None:
                signs = best_signs.copy()
                # Flip 1-3 random edges
                n_flips = rng.integers(1, 4)
                for _ in range(n_flips):
                    idx = rng.integers(0, n_edges)
                    signs[idx] *= -1
            else:
                signs = rng.choice([-1, 1], size=n_edges)
        
        # Make valid
        signs = make_valid_if_needed(signs.copy())
        
        # Local Search with incremental updates
        current_signs = signs.copy()
        current_A = build_signed_adj(current_signs)
        current_radius = spectral_radius_from_A(current_A)
        
        if not is_valid_lift(current_signs):
            continue
            
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            # Try 1-flip
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 1: break
                old = current_signs[i]
                u, v = edges[i]
                # Update A incrementally
                old_val = current_A[u, v]
                current_signs[i] = -old
                current_A[u, v] = -old_val
                current_A[v, u] = -old_val
                
                new_radius = spectral_radius_from_A(current_A)
                if new_radius < current_radius - 1e-9 and is_valid_lift(current_signs):
                    current_radius = new_radius
                    improved = True
                else:
                    # Revert
                    current_signs[i] = old
                    current_A[u, v] = old_val
                    current_A[v, u] = old_val
                    
            # If no 1-flip improved, try 2-flips (stochastic)
            if not improved:
                # Try a few random 2-flips
                for _ in range(20):
                    if time.time() - start_time > DEADLINE - 1: break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    ui, vi = edges[i]
                    uj, vj = edges[j]
                    oi = current_signs[i]
                    oj = current_signs[j]
                    old_ii = current_A[ui, vi]
                    old_jj = current_A[uj, vj]
                    current_signs[i] = -oi
                    current_signs[j] = -oj
                    current_A[ui, vi] = -old_ii
                    current_A[vi, ui] = -old_ii
                    current_A[uj, vj] = -old_jj
                    current_A[vj, uj] = -old_jj
                    
                    new_radius = spectral_radius_from_A(current_A)
                    if new_radius < current_radius - 1e-9 and is_valid_lift(current_signs):
                        current_radius = new_radius
                        improved = True
                    else:
                        # Revert
                        current_signs[i] = oi
                        current_signs[j] = oj
                        current_A[ui, vi] = old_ii
                        current_A[vi, ui] = old_ii
                        current_A[uj, vj] = old_jj
                        current_A[vj, uj] = old_jj
                        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    if best_signs is None:
        # Fallback: find any valid sign assignment
        best_signs = rng.choice([-1, 1], size=n_edges)
        while not is_valid_lift(best_signs):
             # Flip random edge to break balance
             idx = rng.integers(0, n_edges)
             best_signs[idx] *= -1

    return best_signs
