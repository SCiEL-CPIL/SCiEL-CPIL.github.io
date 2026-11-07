"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)
    
    # Build edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Identify edge types: top cycle, bottom cycle, matching
    top_edges = []    # edges within 0..m-1
    bottom_edges = [] # edges within m..2m-1
    match_edges = []  # edges (i, i+m)
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
        elif u >= m and v >= m:
            bottom_edges.append(i)
        else:
            match_edges.append(i)
    
    # Helper: compute spectral radius and check connectivity
    def evaluate(signs):
        # Build signed adjacency matrix
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        # Check 2-lift connectivity
        # Lift vertices: (v, 0) and (v, 1)
        # +1 edge joins (u,0)-(v,0) and (u,1)-(v,1)
        # -1 edge joins (u,0)-(v,1) and (u,1)-(v,0)
        # Build lift adjacency and check connectivity via BFS
        lift_n = 2 * n
        lift_adj = [[] for _ in range(lift_n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[u*2].append(v*2)
                lift_adj[v*2].append(u*2)
                lift_adj[u*2+1].append(v*2+1)
                lift_adj[v*2+1].append(u*2+1)
            else:
                lift_adj[u*2].append(v*2+1)
                lift_adj[v*2+1].append(u*2)
                lift_adj[u*2+1].append(v*2)
                lift_adj[v*2].append(u*2+1)
        
        # BFS from vertex 0
        visited = [False] * lift_n
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        connected = (count == lift_n)
        return radius, connected
    
    # Initial construction based on analytical insights
    # For even m: set top cycle product to -1, bottom cycle product to -1, 
    #   then matching product must be +1 for lift connectivity
    # For odd m: set top cycle product to +1, bottom cycle product to +1,
    #   then matching product must be -1 for lift connectivity
    
    signs = np.ones(E, dtype=np.int8)
    
    if m % 2 == 0:
        # Even m: want top prod = -1, bottom prod = -1, match prod = +1
        # Set one top edge to -1
        if top_edges:
            signs[top_edges[0]] = -1
        # Set one bottom edge to -1
        if bottom_edges:
            signs[bottom_edges[0]] = -1
        # All matching edges +1 (product = +1)
    else:
        # Odd m: want top prod = +1, bottom prod = +1, match prod = -1
        # All top edges +1 (product = +1)
        # All bottom edges +1 (product = +1)
        # Set one matching edge to -1
        if match_edges:
            signs[match_edges[0]] = -1
    
    # Evaluate initial
    best_radius, best_connected = evaluate(signs)
    best_signs = signs.copy()
    
    if not best_connected:
        # Try flipping different edges to fix connectivity
        for i in range(E):
            signs[i] = -signs[i]
            r, c = evaluate(signs)
            signs[i] = -signs[i]
            if c and r < best_radius:
                best_radius = r
                best_signs = signs.copy()
                # Reconstruct from best_signs
                signs[:] = best_signs
    
    # Local search: flip single edges to improve
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(E):
            if time.time() > deadline:
                break
            signs[i] = -signs[i]
            r, c = evaluate(signs)
            signs[i] = -signs[i]
            if c and r < best_radius - 1e-12:
                best_radius = r
                best_signs = signs.copy()
                signs[i] = -signs[i]  # keep the flip
                improved = True
    
    # Try 2-flip moves
    if time.time() < deadline - 5:
        for i in range(E):
            if time.time() > deadline:
                break
            signs[i] = -signs[i]
            for j in range(i+1, E):
                if time.time() > deadline:
                    break
                signs[j] = -signs[j]
                r, c = evaluate(signs)
                signs[j] = signs[j]  # restore
                if c and r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = signs.copy()
                    signs[i] = -signs[i]  # keep flip i
                    signs[j] = -signs[j]  # keep flip j
                    improved = True
                    break
            signs[i] = -signs[i]  # restore
    
    # Final random perturbation search if time remains
    if time.time() < deadline - 3:
        rng = np.random.default_rng(3)
        for _ in range(1000):
            if time.time() > deadline:
                break
            candidate = best_signs.copy()
            # Flip 1-3 random edges
            n_flips = rng.integers(1, 4)
            for _ in range(n_flips):
                idx = rng.integers(0, E)
                candidate[idx] = -candidate[idx]
            r, c = evaluate(candidate)
            if c and r < best_radius - 1e-12:
                best_radius = r
                best_signs = candidate
    
    return best_signs
# EVOLVE-BLOCK-END
