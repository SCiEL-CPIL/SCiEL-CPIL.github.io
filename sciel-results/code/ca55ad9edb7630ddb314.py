import numpy as np
import time

def _build_signed_adjacency(m, signs):
    """
    Build the signed adjacency matrix for the prism graph with given edge signs.
    Prism graph: two m-cycles on vertices 0..m-1 and m..2m-1, joined by matching (i, i+m).
    Edges are ordered: we need to figure out the ordering from graph['edges'].
    """
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip((), signs):
        pass
    # Actually, let's just use the edges from the graph
    return A

def _spectral_radius(m, edges, signs):
    """
    Compute spectral radius of signed adjacency matrix.
    edges: list of (u, v)
    signs: array of +1 or -1, same length as edges
    """
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(edges, signs):
        A[u, v] = float(s)
        A[v, u] = float(s)
    try:
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals))
    except:
        return float('inf')

def _is_connected_lift(m, edges, signs):
    """
    Check if the 2-lift is connected.
    The lift has vertices (v, 0) and (v, 1) for v in 0..2m-1.
    Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
    Edge (u,v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
    
    We check connectivity of the lift graph using BFS.
    """
    n = 2 * m
    # Build adjacency for the lift
    lift_adj = [[] for _ in range(2 * n)]
    for (u, v), s in zip(edges, signs):
        if s > 0:
            # +1: equal sides
            lift_adj[u * 2 + 0].append(v * 2 + 0)
            lift_adj[v * 2 + 0].append(u * 2 + 0)
            lift_adj[u * 2 + 1].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2 + 1)
        else:
            # -1: opposite sides
            lift_adj[u * 2 + 0].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2 + 0)
            lift_adj[u * 2 + 1].append(v * 2 + 0)
            lift_adj[v * 2 + 0].append(u * 2 + 1)
    
    # BFS from (0, 0)
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    while queue:
        node = queue.pop(0)
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return all(visited)

def _eval_signs(m, edges, signs):
    """
    Evaluate a sign assignment. Returns (spectral_radius, is_valid).
    """
    if not _is_connected_lift(m, edges, signs):
        return float('inf'), False
    sr = _spectral_radius(m, edges, signs)
    return sr, True

def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    deadline = time.time() + 55  # 55s deadline with 5s safety margin
    
    # Start with random initial solutions
    rng = np.random.default_rng(42)
    
    # Generate initial random valid solutions
    best_signs = None
    best_sr = float('inf')
    
    # Try random assignments, keep best valid ones
    num_initial = 200
    for i in range(num_initial):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=num_edges)
        sr, valid = _eval_signs(m, edges, signs)
        if valid and sr < best_sr:
            best_sr = sr
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 (should be connected for prism)
        best_signs = np.ones(num_edges, dtype=int)
        best_sr = _spectral_radius(m, edges, best_signs)
    
    # Local search: flip one edge at a time
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(num_edges):
            if time.time() > deadline - 3:
                break
            # Flip edge i
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            sr, valid = _eval_signs(m, edges, best_signs)
            if valid and sr < best_sr:
                best_sr = sr
                improved = True
            else:
                # Revert
                best_signs[i] = old_sign
    
    # Multi-start local search
    num_starts = 50
    for start in range(num_starts):
        if time.time() > deadline - 3:
            break
        
        # Random perturbation of best solution
        signs = best_signs.copy()
        # Flip a few random edges
        num_flips = rng.integers(1, max(2, num_edges // 4))
        flip_indices = rng.choice(num_edges, size=num_flips, replace=False)
        for idx in flip_indices:
            signs[idx] = -signs[idx]
        
        # Local search from this perturbed solution
        local_best_sr, local_valid = _eval_signs(m, edges, signs)
        if not local_valid:
            continue
        
        local_improved = True
        while local_improved and time.time() < deadline - 2:
            local_improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                sr, valid = _eval_signs(m, edges, signs)
                if valid and sr < local_best_sr:
                    local_best_sr = sr
                    local_improved = True
                else:
                    signs[i] = old_sign
        
        if local_best_sr < best_sr:
            best_sr = local_best_sr
            best_signs = signs.copy()
    
    # Final polish: more aggressive local search
    if time.time() < deadline - 1:
        for i in range(num_edges):
            if time.time() > deadline - 1:
                break
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            sr, valid = _eval_signs(m, edges, best_signs)
            if valid and sr < best_sr:
                best_sr = sr
            else:
                best_signs[i] = old_sign
    
    return best_signs
