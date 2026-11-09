import numpy as np
import time
from collections import deque

def _build_base_matrix(n, edges):
    """Build the unsigned adjacency structure (unit weights)."""
    A = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

def _spectral_radius_fast(signs, n, edges, base_A, chord_indices, chord_positions):
    """
    Compute spectral radius efficiently by updating only chord entries.
    signs: array of +1/-1 for all edges
    base_A: the base matrix with all 1s
    chord_indices: list of edge indices that are chords
    chord_positions: for each chord, (u, v) tuple
    """
    A = base_A.copy()
    # Set all edges to +1 first (base_A already has 1s)
    # Now apply signs: for edges with sign -1, negate
    # For tree edges, sign is always +1 in our parameterization
    # For chords, sign depends on the mask
    for k, edge_idx in enumerate(chord_indices):
        if signs[edge_idx] == -1:
            u, v = chord_positions[k]
            A[u, v] = -1.0
            A[v, u] = -1.0
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Build base matrix
    base_A = _build_base_matrix(n, edges)

    # Find spanning tree and identify chords
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    visited = [False] * n
    tree_edge_set = set()
    visited[0] = True
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_set.add(idx)
                dq.append(v)

    chords = [i for i in range(num_edges) if i not in tree_edge_set]
    num_chords = len(chords)
    chord_positions = [edges[i] for i in chords]

    # Precompute: for each chord mask, we need to know which chords are -1
    # Strategy: enumerate all 2^num_chords masks, compute spectral radius
    
    best_radius = float('inf')
    best_mask = 0

    # Phase 1: Exhaustive search over all 2^num_chords chord sign assignments
    # (tree edges fixed at +1, which is a valid parameterization)
    total_masks = 1 << num_chords
    for mask in range(1, total_masks):
        if time.time() > deadline:
            break
        
        # Build signs array
        signs = np.ones(num_edges, dtype=int)
        for k in range(num_chords):
            if mask & (1 << k):
                signs[chords[k]] = -1
        
        r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
        
        if r < best_radius:
            best_radius = r
            best_mask = mask

    # Phase 2: Local search - flip individual chord signs to see if we can improve
    # Start from best_mask and try flipping each chord
    if best_mask > 0:
        current_mask = best_mask
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for k in range(num_chords):
                if time.time() > deadline:
                    break
                # Try flipping chord k
                trial_mask = current_mask ^ (1 << k)
                if trial_mask == 0:
                    continue  # Would be disconnected
                signs = np.ones(num_edges, dtype=int)
                for j in range(num_chords):
                    if trial_mask & (1 << j):
                        signs[chords[j]] = -1
                r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
                if r < best_radius:
                    best_radius = r
                    current_mask = trial_mask
                    best_mask = trial_mask
                    improved = True

    # Phase 3: Random restarts - try random sign assignments (not just tree+chord param)
    # This explores the full space of 2^num_edges sign assignments
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random sign assignment
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        
        # Check connectivity: the 2-lift is connected iff the product of signs
        # around the "fundamental cycle" is not all +1 in a way that disconnects.
        # Actually, for the prism, the lift is connected iff not all matching edges
        # have sign +1 AND the cycle products are compatible.
        # Simpler: just check if the lift is connected by BFS on the lift graph.
        
        # Build lift graph and check connectivity
        # Lift has 2n vertices: (v, 0) and (v, 1)
        # Edge (u,v) with sign s: if s=+1, connects (u,0)-(v,0) and (u,1)-(v,1)
        #                         if s=-1, connects (u,0)-(v,1) and (u,1)-(v,0)
        
        # Quick connectivity check using union-find on the lift
        parent = list(range(2 * n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                union(2*u, 2*v)
                union(2*u+1, 2*v+1)
            else:
                union(2*u, 2*v+1)
                union(2*u+1, 2*v)
        
        # Check if all 2n vertices are in one component
        root = find(0)
        connected = all(find(v) == root for v in range(2 * n))
        
        if connected:
            r = _spectral_radius_fast(signs, n, edges, base_A, chords, chord_positions)
            if r < best_radius:
                best_radius = r
                best_mask = -1  # Mark that we found a non-tree-param solution
                # Store the actual signs
                best_signs = signs.copy()
                break
        if attempts > 50000:
            break

    # If we found a better solution in Phase 3, use it
    if 'best_signs' in locals() and best_mask == -1:
        return best_signs

    # Reconstruct signs from best_mask
    signs = np.ones(num_edges, dtype=int)
    for k in range(num_chords):
        if best_mask & (1 << k):
            signs[chords[k]] = -1
    
    # Final verification: check connectivity
    parent = list(range(2 * n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            union(2*u, 2*v)
            union(2*u+1, 2*v+1)
        else:
            union(2*u, 2*v+1)
            union(2*u+1, 2*v)
    
    root = find(0)
    if not all(find(v) == root for v in range(2 * n)):
        # Fallback: single -1 on first chord
        signs = np.ones(num_edges, dtype=int)
        if chords:
            signs[chords[0]] = -1

    return signs
