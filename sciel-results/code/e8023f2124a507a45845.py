"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_signed_adjacency(graph, signs):
    """Build the n x n signed adjacency matrix."""
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    return S

def _spectral_radius(S):
    """Compute spectral radius of symmetric matrix S."""
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def _is_connected_lift_fast(graph, signs):
    """Fast connectivity check for the 2-lift using union-find with parity."""
    n = graph['n']
    edges = graph['edges']
    
    parent = list(range(n))
    rank = [0] * n
    parity = [0] * n  # parity[i] = x_i XOR x_parent[i]
    
    def find(x):
        if parent[x] != x:
            root = find(parent[x])
            parity[x] ^= parity[parent[x]]
            parent[x] = root
        return parent[x]
    
    def union(x, y, b):
        """Enforce x XOR y = b. Return False if inconsistent."""
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return (parity[x] ^ parity[y]) == b
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
            b = b ^ 1
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        parent[ry] = rx
        parity[ry] = parity[x] ^ parity[y] ^ b
        return True
    
    for i, (u, v) in enumerate(edges):
        b = 0 if signs[i] == 1 else 1
        if not union(u, v, b):
            return True  # Inconsistent -> connected
    
    return False  # Consistent -> disconnected

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    matching_edges.sort()
    
    # Gauge fixing: set matching edges to +1, first top cycle edge to +1
    fixed_signs = np.ones(num_edges, dtype=int)
    for i in matching_edges:
        fixed_signs[i] = 1
    if top_cycle_edges:
        fixed_signs[top_cycle_edges[0]] = 1
    
    free_edges = []
    for i in top_cycle_edges[1:]:
        free_edges.append(i)
    for i in bottom_cycle_edges:
        free_edges.append(i)
    
    num_free = len(free_edges)
    
    best_signs = None
    best_radius = float('inf')
    
    # For small m, do exhaustive search; for larger m, use random + local search
    if num_free <= 20:
        # Exhaustive search over free variables
        for mask in range(1 << num_free):
            if time.time() > deadline:
                break
            
            signs = fixed_signs.copy()
            for j, edge_idx in enumerate(free_edges):
                signs[edge_idx] = -1 if (mask >> j) & 1 else 1
            
            if not _is_connected_lift_fast(graph, signs):
                continue
            
            S = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(S)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Random initialization + local search
        rng = np.random.default_rng(3)
        for attempt in range(min(budget, 200)):
            if time.time() > deadline:
                break
            
            signs = fixed_signs.copy()
            for edge_idx in free_edges:
                signs[edge_idx] = rng.choice([-1, 1])
            
            if not _is_connected_lift_fast(graph, signs):
                continue
            
            S = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(S)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        # Local search refinement
        if best_signs is not None:
            best_signs, best_radius = _local_search(graph, best_signs, deadline)
    
    # If we still don't have a valid solution, try random without gauge fixing
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(budget):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            if _is_connected_lift_fast(graph, signs):
                S = _build_signed_adjacency(graph, signs)
                radius = _spectral_radius(S)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
        
        if best_signs is not None:
            best_signs, best_radius = _local_search(graph, best_signs, deadline)
    
    if best_signs is None:
        # Last resort: all +1 (check if valid, if not try all -1)
        best_signs = np.ones(num_edges, dtype=int)
        if not _is_connected_lift_fast(graph, best_signs):
            best_signs = np.ones(num_edges, dtype=int)
            best_signs[0] = -1  # Flip one edge to try to make it connected
    
    return best_signs

def _local_search(graph, signs, deadline):
    """Local search: flip single edges to reduce spectral radius while maintaining connectivity."""
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    S = _build_signed_adjacency(graph, signs)
    best_radius = _spectral_radius(S)
    best_signs = signs.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            # Try flipping edge i
            old_sign = signs[i]
            signs[i] = -old_sign
            
            # Check connectivity
            if not _is_connected_lift_fast(graph, signs):
                signs[i] = old_sign
                continue
            
            S = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(S)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_sign
    
    return best_signs, best_radius
# EVOLVE-BLOCK-END
