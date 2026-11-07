"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    def build_signed_adj(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_lift_connected(signs):
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        visited = [False] * lift_n
        visited[0] = True
        queue = [0]
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == lift_n
    
    # Classify edges
    top_edges = []
    bot_edges = []
    match_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bot_edges.append(i)
        else:
            match_edges.append(i)
    
    # Generate candidate patterns
    candidates = []
    
    # All positive
    candidates.append(np.ones(n_edges, dtype=int))
    
    # Top +, bottom -, match +
    p = np.ones(n_edges, dtype=int)
    for idx in bot_edges:
        p[idx] = -1
    candidates.append(p)
    
    # Top +, bottom +, match -
    p = np.ones(n_edges, dtype=int)
    for idx in match_edges:
        p[idx] = -1
    candidates.append(p)
    
    # Top -, bottom +, match +
    p = np.ones(n_edges, dtype=int)
    for idx in top_edges:
        p[idx] = -1
    candidates.append(p)
    
    # Alternating on top cycle
    p = np.ones(n_edges, dtype=int)
    for j, idx in enumerate(top_edges):
        p[idx] = 1 if j % 2 == 0 else -1
    candidates.append(p)
    
    # Alternating on bottom cycle
    p = np.ones(n_edges, dtype=int)
    for j, idx in enumerate(bot_edges):
        p[idx] = 1 if j % 2 == 0 else -1
    candidates.append(p)
    
    # Alternating on both cycles
    p = np.ones(n_edges, dtype=int)
    for j, idx in enumerate(top_edges):
        p[idx] = 1 if j % 2 == 0 else -1
    for j, idx in enumerate(bot_edges):
        p[idx] = 1 if j % 2 == 0 else -1
    candidates.append(p)
    
    # Alternating matching
    p = np.ones(n_edges, dtype=int)
    for j, idx in enumerate(match_edges):
        p[idx] = 1 if j % 2 == 0 else -1
    candidates.append(p)
    
    # Top +, bottom -, match alternating
    p = np.ones(n_edges, dtype=int)
    for idx in bot_edges:
        p[idx] = -1
    for j, idx in enumerate(match_edges):
        p[idx] = 1 if j % 2 == 0 else -1
    candidates.append(p)
    
    # Evaluate candidates
    best_sr = float('inf')
    best_signs = None
    for cand in candidates:
        if not is_lift_connected(cand):
            continue
        A = build_signed_adj(cand)
        sr = spectral_radius(A)
        if sr < best_sr:
            best_sr = sr
            best_signs = cand.copy()
    
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
        best_sr = spectral_radius(build_signed_adj(best_signs))
    
    # Local search
    start_time = time.time()
    deadline = start_time + 50.0
    rng = np.random.default_rng(3)
    
    current_signs = best_signs.copy()
    current_sr = best_sr
    no_improve = 0
    
    while time.time() < deadline:
        idx = rng.integers(0, n_edges)
        new_signs = current_signs.copy()
        new_signs[idx] = -new_signs[idx]
        
        A_new = build_signed_adj(new_signs)
        sr_new = spectral_radius(A_new)
        
        if sr_new < current_sr and is_lift_connected(new_signs):
            current_signs = new_signs
            current_sr = sr_new
            if sr_new < best_sr:
                best_sr = sr_new
                best_signs = new_signs.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 500:
            # Random restart
            cand = candidates[rng.integers(0, len(candidates))].copy()
            for _ in range(rng.integers(1, 4)):
                j = rng.integers(0, n_edges)
                cand[j] = -cand[j]
            if is_lift_connected(cand):
                current_signs = cand
                current_sr = spectral_radius(build_signed_adj(cand))
                no_improve = 0
            else:
                no_improve = 0
    
    if not is_lift_connected(best_signs):
        best_signs = np.ones(n_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
