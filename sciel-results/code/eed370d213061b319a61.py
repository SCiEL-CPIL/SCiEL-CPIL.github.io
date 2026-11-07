def sign_edges(graph, budget):
    import numpy as np
    import time
    import random
    
    start_time = time.time()
    DEADLINE = 88.0  # Safety margin before 90s kill
    
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
        
    # Precompute edge indices for faster access? 
    # We need to update the adjacency matrix A.
    # Instead of rebuilding A from scratch, we can update it.
    # But A is small (18x18 max). Rebuilding is fast.
    
    def build_signed_adj(signs):
        n = 2 * m
        A = np.zeros((n, n))
        # Vectorized update? 
        # A[edge_u, edge_v] = signs
        # A[edge_v, edge_u] = signs
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        vals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(vals)))
    
    def is_valid_lift(signs):
        """Check if the 2-lift is connected (unbalanced signed graph)."""
        n = 2 * m
        sigma = np.zeros(n, dtype=int)
        sigma[0] = 1
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        # Use a list as a queue for speed
        q = [0]
        head = 0
        
        while head < len(q):
            u = q[head]
            head += 1
            for edge_idx, v in adj[u]:
                s = signs[edge_idx]
                expected_sigma_v = s * sigma[u]
                if not visited[v]:
                    sigma[v] = expected_sigma_v
                    visited[v] = True
                    q.append(v)
                else:
                    if sigma[v] != expected_sigma_v:
                        return True  # Contradiction: unbalanced, lift is connected
        
        return False  # Balanced: lift is disconnected

    def simulated_annealing(initial_signs, t_start, t_end, steps):
        signs = initial_signs.copy()
        current_radius = spectral_radius(signs)
        if not is_valid_lift(signs):
            return None, current_radius
        
        best_signs = signs.copy()
        best_radius = current_radius
        
        # Coefficient for exponential cooling
        # T = T_start * (T_end / T_start)^(step / steps)
        # Or linear cooling? Exponential is standard.
        
        for step in range(steps):
            if (time.time() - start_time) > DEADLINE:
                break
            
            # Pick a random edge to flip
            i = random.randint(0, n_edges - 1)
            old_sign = signs[i]
            signs[i] = -old_sign
            
            if is_valid_lift(signs):
                new_radius = spectral_radius(signs)
                delta = new_radius - current_radius
                
                if delta < 0:
                    # Accept improvement
                    current_radius = new_radius
                    signs[i] = -signs[i] # Wait, signs[i] is already flipped.
                    # In the code above: signs[i] = -old_sign.
                    # So signs[i] is the new sign.
                    # If we accept, we keep the new sign.
                    # If we reject, we revert.
                    
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = signs.copy()
                else:
                    # Accept worsening with probability exp(-delta / T)
                    # Calculate T
                    if steps == 0:
                        T = t_start
                    else:
                        T = t_start * (t_end / t_start) ** (step / steps)
                    
                    if random.random() < np.exp(-delta / T):
                        current_radius = new_radius
                        # Keep the new sign
                    else:
                        # Revert
                        signs[i] = old_sign
            else:
                # Invalid lift, revert
                signs[i] = old_sign
        
        return best_signs, best_radius

    rng = np.random.default_rng(42)
    best_signs = None
    best_radius = float('inf')
    
    # Initial solutions
    candidates = []
    
    # 1. All +1
    s1 = np.ones(n_edges, dtype=int)
    candidates.append(s1)
    
    # 2. Alternating top
    s2 = np.ones(n_edges, dtype=int)
    top_edges = [i for i, (u, v) in enumerate(edges) if u < m and v < m]
    for i, eidx in enumerate(top_edges):
        if i % 2 == 1:
            s2[eidx] = -1
    candidates.append(s2)
    
    # 3. Alternating bottom
    s3 = np.ones(n_edges, dtype=int)
    bot_edges = [i for i, (u, v) in enumerate(edges) if u >= m and v >= m]
    for i, eidx in enumerate(bot_edges):
        if i % 2 == 1:
            s3[eidx] = -1
    candidates.append(s3)
    
    # 4. Alternating matching
    s4 = np.ones(n_edges, dtype=int)
    match_edges = [i for i, (u, v) in enumerate(edges) if not (u < m and v < m) and not (u >= m and v >= m)]
    for i, eidx in enumerate(match_edges):
        if i % 2 == 1:
            s4[eidx] = -1
    candidates.append(s4)
    
    # 5. Random
    s5 = rng.choice([-1, 1], size=n_edges).astype(int)
    candidates.append(s5)
    
    # Run SA on each candidate
    # We need to distribute time.
    # Let's run a few restarts of SA.
    
    total_steps = 5000 # Estimated steps per restart
    # We will run until deadline.
    
    for restart in range(100):
        if (time.time() - start_time) > DEADLINE - 1:
            break
            
        if restart < len(candidates):
            init = candidates[restart]
        else:
            if best_signs is not None:
                init = best_signs.copy()
                # Perturb
                k = random.randint(1, 3)
                for _ in range(k):
                    idx = random.randint(0, n_edges - 1)
                    init[idx] *= -1
            else:
                init = rng.choice([-1, 1], size=n_edges).astype(int)
        
        s, r = simulated_annealing(init, 1.0, 0.01, total_steps)
        
        if s is not None and r < best_radius:
            best_signs = s
            best_radius = r

    if best_signs is None:
        # Fallback
        best_signs = np.ones(n_edges, dtype=int)
        if len(match_edges) > 0:
            best_signs[match_edges[0]] = -1
    
    return best_signs
