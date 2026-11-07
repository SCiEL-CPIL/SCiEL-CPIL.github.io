"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
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

def _local_search(graph, signs, deadline):
    """
    Local search: flip one edge sign at a time to reduce spectral radius.
    Returns (best_signs, best_radius).
    """
    n_edges = len(graph['edges'])
    signs = np.array(signs, dtype=np.int8)
    best_signs = signs.copy()
    
    # Get initial radius
    valid, best_radius = _is_valid_lift(graph, best_signs)
    if not valid:
        return best_signs, float('inf')
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n_edges):
            if time.time() >= deadline - 2.0:
                break
            # Try flipping sign i
            old_sign = signs[i]
            signs[i] = -signs[i]
            
            valid, radius = _is_valid_lift(graph, signs)
            
            if valid and radius < best_radius - 1e-12:
                best_signs = signs.copy()
                best_radius = radius
                improved = True
            else:
                signs[i] = old_sign
    
    return best_signs, best_radius

def _multi_start_local_search(graph, n_starts, deadline):
    """
    Run multiple random starts followed by local search.
    Returns best (signs, radius) found.
    """
    n_edges = len(graph['edges'])
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # First, try all +1 (often a good starting point)
    signs_all_plus = np.ones(n_edges, dtype=np.int8)
    valid, radius = _is_valid_lift(graph, signs_all_plus)
    if valid and radius < best_radius:
        best_signs = signs_all_plus.copy()
        best_radius = radius
    
    # Try all -1
    signs_all_minus = -np.ones(n_edges, dtype=np.int8)
    valid, radius = _is_valid_lift(graph, signs_all_minus)
    if valid and radius < best_radius:
        best_signs = signs_all_minus.copy()
        best_radius = radius
    
    # Random starts
    for start in range(n_starts):
        if time.time() >= deadline - 5.0:
            break
        
        # Generate random signs
        signs = rng.choice([-1, 1], size=n_edges).astype(np.int8)
        
        # Check validity first
        valid, radius = _is_valid_lift(graph, signs)
        if not valid:
            continue
        
        if radius < best_radius:
            best_signs = signs.copy()
            best_radius = radius
        
        # Local search from this start
        ls_signs, ls_radius = _local_search(graph, signs, deadline)
        if ls_radius < best_radius:
            best_signs = ls_signs
            best_radius = ls_radius
    
    return best_signs, best_radius

def _structured_search(graph, deadline):
    """
    Search over structured sign patterns for prism graphs.
    Prism graph: two m-cycles joined by matching.
    Edges: cycle1 (m edges), cycle2 (m edges), matching (m edges) = 3m edges.
    
    We can exploit symmetry: signs on cycle1, cycle2, and matching.
    """
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    # Identify edge types
    # For prism graphs, edges are typically:
    # - m edges in first cycle: (i, i+1) for i=0..m-1 (with wrap)
    # - m edges in second cycle: (m+i, m+i+1) for i=0..m-1 (with wrap)
    # - m matching edges: (i, m+i) for i=0..m-1
    # But the order may vary, so let's just do a general search.
    
    # Try patterns based on number of -1 signs
    best_signs = None
    best_radius = float('inf')
    
    # Try all combinations of signs for small m
    if m <= 7:
        # Total edges = 3m, so 3m bits. For m=7, that's 21 bits = 2M combos, too many.
        # Instead, try structured patterns.
        pass
    
    # Try patterns where all cycle edges have same sign, matching edges have same sign
    # Pattern: (cycle_sign, matching_sign)
    for cs in [-1, 1]:
        for ms in [-1, 1]:
            signs = np.zeros(n_edges, dtype=np.int8)
            for idx, (u, v) in enumerate(graph['edges']):
                # Determine if this is a matching edge
                # Matching edges connect i to m+i
                if abs(u - v) == m:
                    signs[idx] = ms
                else:
                    signs[idx] = cs
            
            valid, radius = _is_valid_lift(graph, signs)
            if valid and radius < best_radius:
                best_signs = signs.copy()
                best_radius = radius
    
    # Try alternating patterns on cycles
    for cs_cycle1 in [-1, 1]:
        for cs_cycle2 in [-1, 1]:
            for ms in [-1, 1]:
                signs = np.zeros(n_edges, dtype=np.int8)
                for idx, (u, v) in enumerate(graph['edges']):
                    if abs(u - v) == m:
                        signs[idx] = ms
                    elif u < m:
                        # First cycle
                        # Alternating based on vertex index
                        signs[idx] = cs_cycle1 * (1 if u % 2 == 0 else -1)
                    else:
                        # Second cycle
                        v2 = u - m
                        signs[idx] = cs_cycle2 * (1 if v2 % 2 == 0 else -1)
                
                valid, radius = _is_valid_lift(graph, signs)
                if valid and radius < best_radius:
                    best_signs = signs.copy()
                    best_radius = radius
    
    # Local search from best structured pattern
    if best_signs is not None:
        ls_signs, ls_radius = _local_search(graph, best_signs, deadline)
        if ls_radius < best_radius:
            best_signs = ls_signs
            best_radius = ls_radius
    
    return best_signs, best_radius

def sign_edges(graph, budget):
    """
    Returns one sign, -1 or +1, per edge.
    Uses a combination of structured search and local search with multiple starts.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    # Phase 1: Structured search (fast)
    best_signs, best_radius = _structured_search(graph, deadline)
    
    # Phase 2: Multi-start local search with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Determine number of starts based on remaining time
        # Each start + local search takes roughly 0.5-2 seconds for small m
        n_starts = max(1, int(remaining_time / 3.0))
        
        ms_signs, ms_radius = _multi_start_local_search(graph, n_starts, deadline)
        if ms_radius < best_radius:
            best_signs = ms_signs
            best_radius = ms_radius
    
    # Final check: make sure we have a valid result
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(n_edges, dtype=np.int8)
        valid, _ = _is_valid_lift(graph, best_signs)
        if not valid:
            # Try all -1
            best_signs = -np.ones(n_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
