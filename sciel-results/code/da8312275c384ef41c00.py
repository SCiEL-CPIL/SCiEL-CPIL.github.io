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

def _spectral_radius_from_phases(phases, n, edges):
    """Return spectral radius using continuous phases (smooth surrogate)."""
    A = np.zeros((n, n), dtype=complex)
    for i, (u, v) in enumerate(edges):
        w = np.exp(1j * (phases[u] - phases[v]))
        A[u, v] = w
        A[v, u] = np.conj(w)
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def _phase_gradient(phases, n, edges, lr=0.1, steps=200, deadline=None):
    """Gradient descent on phases to minimize spectral radius."""
    phases = phases.copy()
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        # Numerical gradient
        eps = 1e-4
        grad = np.zeros(n)
        for u in range(n):
            phases_plus = phases.copy()
            phases_plus[u] += eps
            phases_minus = phases.copy()
            phases_minus[u] -= eps
            rp = _spectral_radius_from_phases(phases_plus, n, edges)
            rm = _spectral_radius_from_phases(phases_minus, n, edges)
            grad[u] = (rp - rm) / (2 * eps)
        
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            phases -= lr * grad / norm
        
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
    
    return phases

def _phases_to_signs(phases, n, edges):
    """Convert continuous phases to discrete signs."""
    signs = np.ones(len(edges), dtype=int)
    for i, (u, v) in enumerate(edges):
        diff = phases[u] - phases[v]
        if np.cos(diff) < 0:
            signs[i] = -1
    return signs

def _check_connected(signs, n, edges):
    """Check if the 2-lift is connected."""
    n_lift = 2 * n
    adj = [[] for _ in range(n_lift)]
    for i, (u, v) in enumerate(edges):
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
    
    visited = [False] * n_lift
    visited[0] = True
    from collections import deque
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                dq.append(v)
    
    return all(visited)

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

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

    chords = [i for i in range(num_edges) if i not in tree_edge_set]
    num_chords = len(chords)

    best_radius = float('inf')
    best_signs = None

    # Phase 1: Exhaustive search over chord assignments
    for mask in range(1, 1 << num_chords):
        if time.time() > deadline:
            break
        
        signs = np.ones(num_edges, dtype=int)
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
        
        if not _check_connected(signs, n, edges):
            continue
            
        r = _spectral_radius(signs, n, edges)
        
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Phase 2: Continuous phase optimization
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline:
                break
            
            phases = rng.uniform(-np.pi, np.pi, size=n)
            phases_opt = _phase_gradient(phases, n, edges, lr=0.5, steps=80, deadline=deadline)
            signs = _phases_to_signs(phases_opt, n, edges)
            
            if not _check_connected(signs, n, edges):
                continue
            
            r = _spectral_radius(signs, n, edges)
            
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        
        if best_signs is not None:
            phases_struct = np.zeros(n)
            visited_phase = [False] * n
            visited_phase[0] = True
            dq = deque([0])
            while dq:
                u = dq.popleft()
                for v, idx in adj[u]:
                    if not visited_phase[v]:
                        visited_phase[v] = True
                        s = best_signs[idx]
                        if s == 1:
                            phases_struct[v] = phases_struct[u]
                        else:
                            phases_struct[v] = phases_struct[u] + np.pi
                        dq.append(v)
            
            phases_opt = _phase_gradient(phases_struct, n, edges, lr=0.3, steps=100, deadline=deadline)
            signs = _phases_to_signs(phases_opt, n, edges)
            
            if _check_connected(signs, n, edges):
                r = _spectral_radius(signs, n, edges)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()

    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if chords:
            signs[chords[0]] = -1
        best_signs = signs

    return best_signs
