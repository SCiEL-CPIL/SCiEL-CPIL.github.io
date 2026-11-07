def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 85
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Classify edges
    edge_type = np.zeros(n_edges, dtype=np.int32)
    top_indices = []
    bottom_indices = []
    match_indices = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            edge_type[i] = 0
            top_indices.append(i)
        elif u >= m and v >= m:
            edge_type[i] = 1
            bottom_indices.append(i)
        else:
            edge_type[i] = 2
            match_indices.append(i)
    
    top_indices = np.array(top_indices)
    bottom_indices = np.array(bottom_indices)
    match_indices = np.array(match_indices)
    
    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius(A):
        return np.max(np.abs(np.linalg.eigvalsh(A)))
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected using BFS."""
        # Build adjacency of the lift
        lift_n = 2 * n_vertices
        adj = [[] for _ in range(lift_n)]
        for i in range(n_edges):
            u, v = int(edge_u[i]), int(edge_v[i])
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
        
        visited = np.zeros(lift_n, dtype=bool)
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    if count == lift_n:
                        return True
                    stack.append(nb)
        return count == lift_n
    
    def is_connected_lift_fast(signs):
        """Faster connectivity check using the cycle product criterion."""
        # The lift is disconnected iff the signing is balanced.
        # A signing is balanced iff there exists s:V->{±1} with σ(u,v)=s(u)s(v).
        # This is equivalent to: for every cycle, product of signs = +1.
        # For prism: check top cycle product, bottom cycle product, and 
        # for each i: σ(i,i+1)*σ(i+1,m+i+1)*σ(m+i+1,m+i)*σ(m+i,i) = +1
        # Actually, it's sufficient to check: the product around the top cycle 
        # and the product around one "ladder" cycle.
        
        # Simpler: do BFS but with early exit
        lift_n = 2 * n_vertices
        adj = [[] for _ in range(lift_n)]
        for i in range(n_edges):
            u, v = int(edge_u[i]), int(edge_v[i])
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
        
        visited = np.zeros(lift_n, dtype=bool)
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    if count == lift_n:
                        return True
                    stack.append(nb)
        return count == lift_n
    
    def evaluate(signs):
        """Evaluate a signing: return (radius, is_valid)."""
        if not is_connected_lift_fast(signs):
            return float('inf'), False
        A = build_signed_adj(signs)
        radius = spectral_radius(A)
        return radius, True
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase 1: Try structured patterns
    # For prism graphs, try all 8 combinations of constant signs per edge type
    for top_s in [-1, 1]:
        for bot_s in [-1, 1]:
            for match_s in [-1, 1]:
                signs = np.ones(n_edges, dtype=int)
                signs[top_indices] = top_s
                signs[bottom_indices] = bot_s
                signs[match_indices] = match_s
                radius, valid = evaluate(signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    # Phase 2: Simulated annealing
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_A = build_signed_adj(current_signs)
        current_radius = best_radius
    else:
        # Find any valid starting point
        current_signs = rng.choice([-1, 1], size=n_edges)
        for _ in range(10000):
            if is_connected_lift_fast(current_signs):
                break
            idx = rng.integers(0, n_edges)
            current_signs[idx] *= -1
        current_A = build_signed_adj(current_signs)
        current_radius = spectral_radius(current_A)
        best_radius = current_radius
        best_signs = current_signs.copy()
    
    T_start = 1.0
    T_end = 0.01
    n_sa_steps = 50000
    
    for step in range(n_sa_steps):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_sa_steps)
        
        # Pick a random edge to flip
        i = rng.integers(0, n_edges)
        old_sign = current_signs[i]
        new_sign = -old_sign
        current_signs[i] = new_sign
        
        u, v = int(edge_u[i]), int(edge_v[i])
        current_A[u, v] = new_sign
        current_A[v, u] = new_sign
        
        new_radius = spectral_radius(current_A)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        else:
            # Accept with probability exp(-(new_radius - current_radius)/T)
            delta = new_radius - current_radius
            if delta < 0:
                pass
            else:
                if rng.random() < np.exp(-delta / T):
                    current_radius = new_radius
                else:
                    # Reject
                    current_signs[i] = old_sign
                    current_A[u, v] = old_sign
                    current_A[v, u] = old_sign
    
    # Phase 3: Local search from best
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_A = build_signed_adj(current_signs)
        current_radius = best_radius
        
        improved = True
        max_iters = 20
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 2:
                break
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 2:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                u, v = int(edge_u[i]), int(edge_v[i])
                current_A[u, v] = current_signs[i]
                current_A[v, u] = current_signs[i]
                
                new_radius = spectral_radius(current_A)
                
                if new_radius < current_radius - 1e-9 and is_connected_lift_fast(current_signs):
                    current_radius = new_radius
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old
                    current_A[u, v] = old
                    current_A[v, u] = old
    
    # Phase 4: More random restarts with short local search
    for restart in range(200):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        if best_signs is not None:
            signs = best_signs.copy()
            n_flips = rng.integers(2, 8)
            idxs = rng.choice(n_edges, size=n_flips, replace=False)
            signs[idxs] *= -1
        else:
            signs = rng.choice([-1, 1], size=n_edges)
        
        if not is_connected_lift_fast(signs):
            continue
        
        current_signs = signs.copy()
        current_A = build_signed_adj(current_signs)
        current_radius = spectral_radius(current_A)
        
        # Short local search
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 3:
                break
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 3:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                u, v = int(edge_u[i]), int(edge_v[i])
                current_A[u, v] = current_signs[i]
                current_A[v, u] = current_signs[i]
                
                new_radius = spectral_radius(current_A)
                
                if new_radius < current_radius - 1e-9 and is_connected_lift_fast(current_signs):
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old
                    current_A[u, v] = old
                    current_A[v, u] = old
            
            if not improved:
                break
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
    
    if best_signs is None:
        best_signs = rng.choice([-1, 1], size=n_edges)
        for _ in range(10000):
            if is_connected_lift_fast(best_signs):
                break
            idx = rng.integers(0, n_edges)
            best_signs[idx] *= -1
    
    return best_signs
