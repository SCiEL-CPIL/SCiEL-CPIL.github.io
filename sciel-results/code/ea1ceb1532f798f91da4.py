"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_adjacency(graph):
    """Build the signed adjacency matrix structure."""
    n = graph['n']
    edges = graph['edges']
    # Store edge indices for quick lookup
    return n, edges

def _signed_adj_matrix(n, edges, signs):
    """Build the signed adjacency matrix."""
    A = np.zeros((n, n))
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _is_valid_lift(n, edges, signs):
    """Check if the lift is connected and 3-regular."""
    # The lift has 2n vertices
    # Each vertex (v, 0) and (v, 1)
    # For each edge (u, v) with sign s:
    #   if s = +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   if s = -1: (u,0)-(v,1) and (u,1)-(v,0)
    
    # Build adjacency for the lift
    lift_adj = {}
    for i in range(2 * n):
        lift_adj[i] = []
    
    for (u, v), s in zip(edges, signs):
        # (u, 0) -> (v, 0) if s=+1, (v, 1) if s=-1
        # (u, 1) -> (v, 1) if s=+1, (v, 0) if s=-1
        # And symmetrically
        if s == 1:
            lift_adj[u].append(v)
            lift_adj[n + u].append(n + v)
            lift_adj[v].append(u)
            lift_adj[n + v].append(n + u)
        else:
            lift_adj[u].append(n + v)
            lift_adj[n + u].append(v)
            lift_adj[v].append(n + u)
            lift_adj[n + v].append(u)
    
    # Check 3-regular: each vertex has exactly 3 neighbors
    for i in range(2 * n):
        if len(lift_adj[i]) != 3:
            return False
    
    # Check connectivity using BFS
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        node = queue.pop(0)
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == 2 * n

def _spectral_radius(n, edges, signs):
    """Compute the spectral radius of the signed adjacency matrix."""
    A = _signed_adj_matrix(n, edges, signs)
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def sign_edges(graph, budget):
    """Find signs for edges to minimize the spectral radius of the signed adjacency matrix."""
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second deadline with safety margin
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types: cycle edges and matching edges
    # Cycle 1: vertices 0..m-1
    # Cycle 2: vertices m..2m-1
    # Matching: (i, i+m) for i in 0..m-1
    
    cycle1_edges = []
    cycle2_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle1_edges.append(idx)
        elif u >= m and v >= m:
            cycle2_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Strategy: Use simulated annealing or local search
    # Start with a good initial configuration
    
    best_signs = None
    best_radius = np.inf
    
    # Try a few random starts and do local search
    rng = np.random.default_rng(3)
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Check validity and compute radius
        if _is_valid_lift(n, edges, signs):
            radius = _spectral_radius(n, edges, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        # Local search: flip one edge at a time
        current_signs = signs.copy()
        current_radius = best_radius if (best_signs is not None and np.array_equal(current_signs, best_signs)) else _spectral_radius(n, edges, current_signs) if _is_valid_lift(n, edges, current_signs) else np.inf
        
        # If current is invalid, try to fix it
        if not _is_valid_lift(n, edges, current_signs):
            # Try flipping each edge to see if we can get a valid lift
            found_valid = False
            for i in range(num_edges):
                trial = current_signs.copy()
                trial[i] *= -1
                if _is_valid_lift(n, edges, trial):
                    current_signs = trial
                    current_radius = _spectral_radius(n, edges, current_signs)
                    found_valid = True
                    break
            if not found_valid:
                continue
        
        # Now do local search
        for step in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Try flipping a random edge
            i = rng.integers(0, num_edges)
            trial = current_signs.copy()
            trial[i] *= -1
            
            if _is_valid_lift(n, edges, trial):
                trial_radius = _spectral_radius(n, edges, trial)
                if trial_radius < current_radius:
                    current_signs = trial
                    current_radius = trial_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
        
        # Also try systematic local search: flip each edge once
        for i in range(num_edges):
            if time.time() > deadline - 5.0:
                break
            
            trial = current_signs.copy()
            trial[i] *= -1
            
            if _is_valid_lift(n, edges, trial):
                trial_radius = _spectral_radius(n, edges, trial)
                if trial_radius < current_radius:
                    current_signs = trial
                    current_radius = trial_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 signs (should be valid for prism graphs)
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
