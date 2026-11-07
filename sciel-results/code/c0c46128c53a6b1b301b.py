"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_signed_matrix(n, edges, signs):
    """Build the signed adjacency matrix."""
    m = n * n
    S = np.zeros((n, n), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        val = signs[i]
        S[u, v] = val
        S[v, u] = val
    return S

def _spectral_radius_signed(S):
    """Compute spectral radius of signed adjacency matrix."""
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def _check_connectivity_lift(n, edges, signs):
    """
    Check if the 2-lift is connected.
    The lift has vertices (v, 0) and (v, 1) for v in 0..n-1.
    Edge (u, v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
    Edge (u, v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
    """
    # Build adjacency list for the lift
    # Total vertices in lift: 2*n
    # Vertex (v, s) is indexed as v * 2 + s
    lift_adj = [[] for _ in range(2 * n)]
    
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            # Connect equal sides
            lift_adj[u * 2].append(v * 2)
            lift_adj[v * 2].append(u * 2)
            lift_adj[u * 2 + 1].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2 + 1)
        else:
            # Connect opposite sides
            lift_adj[u * 2].append(v * 2 + 1)
            lift_adj[v * 2 + 1].append(u * 2)
            lift_adj[u * 2 + 1].append(v * 2)
            lift_adj[v * 2].append(u * 2 + 1)
    
    # BFS/DFS to check connectivity
    visited = [False] * (2 * n)
    stack = [0]
    visited[0] = True
    count = 1
    
    while stack:
        node = stack.pop()
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                stack.append(neighbor)
    
    return count == 2 * n

def _is_valid_lift(n, edges, signs):
    """Check if the lift is valid (connected and 3-regular)."""
    # The base graph is 3-regular (prism graph), so the lift is automatically 3-regular
    # if the base is. We just need to check connectivity.
    return _check_connectivity_lift(n, edges, signs)

def _compute_radius(n, edges, signs):
    """Compute spectral radius, return None if invalid."""
    if not _is_valid_lift(n, edges, signs):
        return None
    S = _build_signed_matrix(n, edges, signs)
    return _spectral_radius_signed(S)

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Local search with 1-flip and 2-flip moves, plus random restarts.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute edge indices for quick lookup
    # We need to know which edges are cycle edges vs matching edges
    # For prism graph: two m-cycles (0..m-1 and m..2m-1) and matching (i, i+m)
    # m = n // 2
    m = n // 2
    
    # Identify edge types
    # Cycle edges: (i, i+1) mod m for first cycle, (m+i, m+i+1) mod m for second
    # Matching edges: (i, i+m)
    
    # Precompute adjacency structure for faster local search
    # For each edge, know its endpoints
    edge_u = np.array([e[0] for e in edges])
    edge_v = np.array([e[1] for e in edges])
    
    # Build a map from edge index to position
    # Also, for local search, we want to quickly evaluate the effect of flipping
    
    # Initial best solution from seed
    rng = np.random.default_rng(3)
    
    # Generate initial candidates
    best_signs = None
    best_radius = float('inf')
    
    # Sample budget random sign vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        radius = _compute_radius(n, edges, signs)
        if radius is not None and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=np.int8)
        best_radius = _compute_radius(n, edges, best_signs)
    
    # Local search
    while time.time() < deadline:
        # Try 1-flip moves
        improved = False
        for i in range(num_edges):
            if time.time() > deadline - 5:
                break
            new_signs = best_signs.copy()
            new_signs[i] = -new_signs[i]
            radius = _compute_radius(n, edges, new_signs)
            if radius is not None and radius < best_radius:
                best_signs = new_signs
                best_radius = radius
                improved = True
                break  # Restart after improvement
        
        if not improved:
            # Try 2-flip moves
            improved_2 = False
            # Sample some 2-flip pairs
            num_pairs = min(100, num_edges * (num_edges - 1) // 2)
            pair_indices = rng.choice(num_edges * (num_edges - 1) // 2, size=num_pairs, replace=False)
            for pi in pair_indices:
                if time.time() > deadline - 5:
                    break
                # Convert linear index to pair (i, j) with i < j
                # pi = i * (num_edges - 1) - i * (i + 1) // 2 + (j - i - 1)
                # Simpler: just generate random pairs
                i = rng.integers(0, num_edges)
                j = rng.integers(0, num_edges)
                if i == j:
                    continue
                new_signs = best_signs.copy()
                new_signs[i] = -new_signs[i]
                new_signs[j] = -new_signs[j]
                radius = _compute_radius(n, edges, new_signs)
                if radius is not None and radius < best_radius:
                    best_signs = new_signs
                    best_radius = radius
                    improved_2 = True
                    break
            
            if not improved_2:
                # Random restart
                signs = rng.choice([-1, 1], size=num_edges)
                radius = _compute_radius(n, edges, signs)
                if radius is not None and radius < best_radius:
                    best_signs = signs.copy()
                    best_radius = radius
    
    return best_signs
# EVOLVE-BLOCK-END
