import numpy as np
import time
from collections import deque

def _lift_connected_and_radius(signs, n, edges, m):
    """Return spectral radius if lift is connected 3-regular, else None."""
    N = 2 * n
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            pairs = [(2*u, 2*v), (2*u+1, 2*v+1)]
        else:
            pairs = [(2*u, 2*v+1), (2*u+1, 2*v)]
        for a, b in pairs:
            adj[a].append(b)
            adj[b].append(a)
    seen = [False] * N
    seen[0] = True
    stack = [0]
    count = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                count += 1
                stack.append(y)
    if count != N:
        return None
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        A[u, v] = s
        A[v, u] = s
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Build adjacency for spanning tree
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

    chords = [i for i in range(E) if i not in tree_edge_set]
    num_chords = len(chords)

    best_radius = float('inf')
    best_signs = None

    def try_update(signs):
        nonlocal best_radius, best_signs
        r = _lift_connected_and_radius(signs, n, edges, m)
        if r is not None and r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            return True
        return False

    # 1. Structured Enumeration (Tree edges fixed to +1)
    # This is fast: 2^(m+1)
    # Find a valid base: set first chord to -1
    base_signs = np.ones(E, dtype=int)
    if num_chords > 0:
        base_signs[chords[0]] = -1
    try_update(base_signs)

    # Enumerate all chord combinations
    for mask in range(1 << num_chords):
        if time.time() > deadline - 10:
            break
        signs = np.ones(E, dtype=int)
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
        try_update(signs)

    # 2. Local Search from Best Structured Candidate and Random Candidates
    candidates = []
    if best_signs is not None:
        candidates.append(best_signs)

    rng = np.random.default_rng(3)
    # Generate random valid candidates
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=E)
        # Check if valid
        r = _lift_connected_and_radius(signs, n, edges, m)
        if r is not None:
            candidates.append(signs)

    # Run single-flip local search on each candidate
    for signs in candidates:
        if time.time() > deadline - 3:
            break
        
        current_signs = signs.copy()
        current_radius = _lift_connected_and_radius(current_signs, n, edges, m)
        if current_radius is None:
            continue
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(E):
                if time.time() > deadline - 3:
                    break
                new_signs = current_signs.copy()
                new_signs[i] *= -1
                new_r = _lift_connected_and_radius(new_signs, n, edges, m)
                if new_r is not None and new_r < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = new_r
                    improved = True
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    # 3. Pairwise Polish on Global Best
    if best_signs is not None and time.time() < deadline - 1:
        current_signs = best_signs.copy()
        current_radius = best_radius
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(E):
                if time.time() > deadline - 1:
                    break
                for j in range(i + 1, E):
                    if time.time() > deadline - 1:
                        break
                    new_signs = current_signs.copy()
                    new_signs[i] *= -1
                    new_signs[j] *= -1
                    new_r = _lift_connected_and_radius(new_signs, n, edges, m)
                    if new_r is not None and new_r < current_radius - 1e-10:
                        current_signs = new_signs
                        current_radius = new_r
                        improved = True
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    if best_signs is None:
        # Fallback
        best_signs = base_signs

    return best_signs
