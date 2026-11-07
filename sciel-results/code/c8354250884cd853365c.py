"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix from graph and signs."""
    n = graph['n']
    A = np.zeros((n, n))
    for edge, s in zip(graph['edges'], signs):
        u, v = edge
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius of symmetric matrix."""
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _local_search(graph, signs, max_iters=500):
    """Local search: flip edges to reduce spectral radius."""
    n_edges = len(signs)
    signs = signs.copy()
    A = _build_signed_adjacency(graph, signs)
    current_radius = _spectral_radius(A)
    
    # Verify validity
    valid, _ = _is_valid_lift(graph, signs)
    if not valid:
        return signs, current_radius
    
    for it in range(max_iters):
        improved = False
        for i in range(n_edges):
            # Try flipping edge i
            old_sign = signs[i]
            signs[i] = -old_sign
            
            A = _build_signed_adjacency(graph, signs)
            new_radius = _spectral_radius(A)
            
            # Check validity
            valid, _ = _is_valid_lift(graph, signs)
            
            if valid and new_radius < current_radius:
                current_radius = new_radius
                improved = True
            else:
                signs[i] = old_sign
        
        if not improved:
            break
    
    return signs, current_radius

def _init_alternating(graph, m):
    """Initialize with alternating signs on cycles, +1 on matching."""
    signs = []
    # Top cycle: edges between 0-1, 1-2, ..., (m-1)-0
    # Bottom cycle: edges between m-(m+1), ..., (2m-1)-m
    # Matching: i-(i+m)
    
    # We need to figure out the edge order from the graph
    # Let's identify edges by type
    for (u, v) in graph['edges']:
        if u < m and v < m:
            # Top cycle edge
            # Alternate signs
            signs.append(1 if (u % 2 == 0) else -1)
        elif u >= m and v >= m:
            # Bottom cycle edge
            signs.append(1 if ((u - m) % 2 == 0) else -1)
        else:
            # Matching edge
            signs.append(1)
    
    return np.array(signs, dtype=int)

def _init_random(graph, rng):
    """Random initialization."""
    n_edges = len(graph['edges'])
    return rng.choice([-1, 1], size=n_edges)

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try multiple initializations and local searches
    num_restarts = min(budget, 20)
    
    for restart in range(num_restarts):
        if restart == 0:
            # Start with alternating pattern
            signs = _init_alternating(graph, m)
        elif restart == 1:
            # Start with all +1
            signs = np.ones(n_edges, dtype=int)
        elif restart == 2:
            # Start with all -1 on matching, alternating on cycles
            signs = _init_alternating(graph, m).copy()
            # Flip matching signs
            for i, (u, v) in enumerate(graph['edges']):
                if not (u < m and v < m) and not (u >= m and v >= m):
                    signs[i] = -1
        else:
            # Random initialization
            signs = _init_random(graph, rng)
        
        # Check if valid first
        valid, radius = _is_valid_lift(graph, signs)
        if not valid:
            # Try to fix by flipping one edge
            fixed = False
            for i in range(n_edges):
                signs[i] = -signs[i]
                valid, radius = _is_valid_lift(graph, signs)
                if valid:
                    fixed = True
                    break
                signs[i] = -signs[i]
            if not fixed:
                continue
        
        # Local search
        signs, radius = _local_search(graph, signs, max_iters=100)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: use the seed approach
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
    
    return best_signs
# EVOLVE-BLOCK-END
