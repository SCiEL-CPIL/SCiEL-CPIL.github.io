"""Sign-edges solver for prism-graph spectral lifts: enumeration + local search."""
import time
import numpy as np

_DEADLINE = 85.0


def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        sv = float(s)
        A[u, v] = sv
        A[v, u] = sv
    return A


def _is_coboundary_fast(graph, signs):
    """Check if sign assignment is a coboundary using BFS."""
    n = graph['n']
    edges = graph['edges']
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    f = [-1] * n
    f[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v, idx in adj[u]:
            s = signs[idx]
            expected_fv = f[u] + (0 if s == 1 else 1)
            if f[v] == -1:
                f[v] = expected_fv % 2
                stack.append(v)
            else:
                if f[v] != expected_fv % 2:
                    return False
    return True


def _lift_radius(graph, signs):
    """Return spectral radius if lift is connected, else inf."""
    if _is_coboundary_fast(graph, signs):
        return float('inf')
    A = _build_signed_adj(graph, signs)
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float('inf')


def _local_search(graph, signs, radius, deadline):
    """Local search with single and double flips."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius = radius
    best_signs = signs.copy()
    
    improved = True
    stall_count = 0
    while improved and time.time() < deadline:
        improved = False
        
        # Single flips
        for i in range(n_edges):
            if time.time() >= deadline:
                return best_signs, best_radius
            signs[i] = -signs[i]
            r = _lift_radius(graph, signs)
            if r < best_radius - 1e-12:
                best_radius = r
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = -signs[i]
        if improved:
            stall_count = 0
            continue
        stall_count += 1
        
        # Double flips
        if time.time() >= deadline:
            return best_signs, best_radius
        for i in range(n_edges):
            if time.time() >= deadline:
                return best_signs, best_radius
            for j in range(i + 1, n_edges):
                if time.time() >= deadline:
                    return best_signs, best_radius
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                r = _lift_radius(graph, signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    break
                else:
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
            if improved:
                break
        
        # Triple flips if stuck
        if not improved and stall_count >= 2 and time.time() < deadline:
            for i in range(n_edges):
                if time.time() >= deadline:
                    return best_signs, best_radius
                for j in range(i + 1, n_edges):
                    if time.time() >= deadline:
                        return best_signs, best_radius
                    for k in range(j + 1, min(j + 6, n_edges)):
                        if time.time() >= deadline:
                            return best_signs, best_radius
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        signs[k] = -signs[k]
                        r = _lift_radius(graph, signs)
                        if r < best_radius - 1e-12:
                            best_radius = r
                            best_signs = signs.copy()
                            improved = True
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            signs[k] = -signs[k]
                            break
                        else:
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            signs[k] = -signs[k]
                    if improved:
                        break
                if improved:
                    break
            stall_count = 0
    
    return best_signs, best_radius


def sign_edges(graph, budget):
    start = time.time()
    deadline = start + _DEADLINE
    
    n = graph['n']
    m = n // 2
    n_edges = len(graph['edges'])
    
    if n_edges == 0:
        return np.array([], dtype=np.int64)
    
    # For small m, try enumeration
    if m <= 5 and n_edges <= 16:
        # Enumerate all sign assignments
        best_signs = None
        best_radius = float('inf')
        total = 2 ** n_edges
        for mask in range(total):
            if time.time() >= deadline:
                break
            signs = np.zeros(n_edges, dtype=np.int64)
            for i in range(n_edges):
                signs[i] = 1 if (mask >> i) & 1 else -1
            r = _lift_radius(graph, signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        
        if best_signs is not None:
            return np.asarray(best_signs, dtype=np.int64)
    
    # For larger m, use local search with multiple restarts
    inits = []
    
    # Structured patterns
    inits.append(np.ones(n_edges, dtype=np.int64))
    inits.append(-np.ones(n_edges, dtype=np.int64))
    
    # Alternating on bottom cycle
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    # Alternating on top cycle
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[m + i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    # Alternating on matching
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[2*m + i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    # Single -1 on each edge
    for i in range(n_edges):
        signs = np.ones(n_edges, dtype=np.int64)
        signs[i] = -1
        inits.append(signs.copy())
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(30):
        signs = rng.choice([-1, 1], size=n_edges)
        inits.append(signs)
    
    # Evaluate all inits, keep top few
    candidates = []
    for init in inits:
        if time.time() >= deadline:
            break
        r = _lift_radius(graph, init)
        if np.isfinite(r):
            candidates.append((r, init.copy()))
    
    if not candidates:
        rng = np.random.default_rng(3)
        for _ in range(min(budget, 50)):
            if time.time() >= deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges)
            r = _lift_radius(graph, signs)
            if np.isfinite(r):
                candidates.append((r, signs.copy()))
    
    if not candidates:
        return np.ones(n_edges, dtype=np.int64)
    
    # Sort by radius
    candidates.sort(key=lambda x: x[0])
    
    # Local search from top candidates
    best_signs = None
    best_radius = float('inf')
    
    num_restarts = min(5, len(candidates))
    for r, init in candidates[:num_restarts]:
        if time.time() >= deadline:
            break
        s, rad = _local_search(graph, init, r, deadline)
        if rad < best_radius:
            best_radius = rad
            best_signs = s
    
    if best_signs is None:
        return np.ones(n_edges, dtype=np.int64)
    
    return np.asarray(best_signs, dtype=np.int64)
