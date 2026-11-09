import numpy as np
import time

def _spectral_radius(signs, n, edges):
    """Return spectral radius of signed adjacency matrix."""
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        A[u, v] = s
        A[v, u] = s
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2

    # Identify chords (non-tree edges)
    # For a prism graph, we can easily identify them.
    # Edges are typically ordered: cycle 1, cycle 2, matching.
    # Or just use BFS to find a spanning tree.
    
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    visited = [False] * n
    tree_edge_set = set()
    visited[0] = True
    from collections import deque
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_set.add(idx)
                dq.append(v)

    chords = [i for i in range(len(edges)) if i not in tree_edge_set]
    num_chords = len(chords)

    best_radius = float('inf')
    best_signs = None

    # Enumerate all 2^num_chords assignments
    # Exclude the all-+1 assignment (disconnected)
    for mask in range(1, 1 << num_chords):
        if time.time() > deadline:
            break
        
        # Construct signs
        # Start with all +1
        # Set chords based on mask
        # Note: tree edges are always +1 in this parameterization
        
        # Optimization: Instead of creating new array each time, 
        # we can update the chord positions.
        # But 1024 iterations is small.
        
        signs = np.ones(len(edges), dtype=int)
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
        
        # Connectivity check:
        # Disconnected iff all chord signs are +1.
        # Since mask > 0, at least one bit is set, so at least one chord is -1.
        # So all these assignments are connected.
        
        r = _spectral_radius(signs, n, edges)
        
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # If no valid signs found (should not happen for m>=4), fallback
    if best_signs is None:
        # Fallback: single -1 on first chord
        signs = np.ones(len(edges), dtype=int)
        if chords:
            signs[chords[0]] = -1
        best_signs = signs

    return best_signs
