import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Find spanning tree using BFS
    visited = [False] * n
    tree_edge_indices = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, edge_idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_indices.add(edge_idx)
                queue.append(v)
                
    # Identify chords (edges not in tree)
    chords = []
    for i in range(num_edges):
        if i not in tree_edge_indices:
            chords.append(i)
            
    num_chords = len(chords)
    
    # Precompute base matrix with all +1 (tree edges fixed at +1)
    base_A = np.ones((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        if i in tree_edge_indices:
            base_A[u, v] = 1.0
            base_A[v, u] = 1.0
        else:
            # Chord edges will be set dynamically
            base_A[u, v] = 0.0  # placeholder
            base_A[v, u] = 0.0
    
    # Store chord endpoints for quick updating
    chord_endpoints = [(edges[i][0], edges[i][1]) for i in chords]
    
    best_radius = float('inf')
    best_signs = None
    
    # Iterate over all masks
    # Mask bit k corresponds to chords[k]
    # 0 -> +1, 1 -> -1
    # We skip mask 0 because it results in a balanced (disconnected lift) graph
    for mask in range(1, 1 << num_chords):
        if time.time() > deadline:
            if best_signs is not None:
                return best_signs
            # Fallback to a valid sign pattern if no time left and nothing found
            signs = np.ones(num_edges, dtype=int)
            signs[chords[0]] = -1
            return signs
                
        signs = np.ones(num_edges, dtype=int)
        
        # Assign signs to chords
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
                
        # Build signed adjacency matrix by copying base and setting chords
        A = base_A.copy()
        for k, (u, v) in enumerate(chord_endpoints):
            s = 1.0 if (mask & (1 << k)) == 0 else -1.0
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is not None:
        return best_signs
    
    # Final fallback
    signs = np.ones(num_edges, dtype=int)
    signs[chords[0]] = -1
    return signs
