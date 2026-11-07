import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = graph['n'] // 2
    n = graph['n']
    edges = graph['edges']
    n_edges = len(edges)
    
    # Build adjacency list for spanning tree
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Find spanning tree using DFS
    visited = [False] * n
    tree_edges = set()
    
    def dfs(u):
        visited[u] = True
        for v, idx in adj[u]:
            if not visited[v]:
                tree_edges.add(idx)
                dfs(v)
                
    dfs(0)
    
    # Identify non-tree edges
    non_tree_indices = []
    for i in range(n_edges):
        if i not in tree_edges:
            non_tree_indices.append(i)
            
    k = len(non_tree_indices)
    
    # Precompute edge endpoints for faster matrix construction
    edge_u = [e[0] for e in edges]
    edge_v = [e[1] for e in edges]
    
    best_radius = float('inf')
    best_signs = np.ones(n_edges, dtype=int)
    
    # Iterate over all 2^k combinations of signs for non-tree edges
    # Tree edges are fixed to +1
    for mask in range(1 << k):
        if time.time() > deadline:
            break
            
        # Check connectivity condition:
        # Lift is connected iff at least one non-tree edge has sign -1.
        # In our mask, bit j=1 means sign -1, bit j=0 means sign +1.
        # So we skip mask == 0 (all +1).
        if mask == 0:
            continue
            
        # Construct signs array
        signs = np.ones(n_edges, dtype=int)
        for j, idx in enumerate(non_tree_indices):
            if (mask >> j) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Construct signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for i in range(n_edges):
            s = float(signs[i])
            u = edge_u[i]
            v = edge_v[i]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigenvalues))
        except np.linalg.LinAlgError:
            continue
            
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Fallback if no valid lift found (should not happen as k>=1 for m>=2)
    if best_radius == float('inf'):
        # Simple heuristic fallback
        signs = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                top = u if u < m else v
                if top % 2 == 0:
                    signs[i] = -1
        return signs
        
    return best_signs
