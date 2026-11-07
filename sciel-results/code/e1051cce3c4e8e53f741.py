"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    
    # Build adjacency structure
    # Map edge index to (u, v)
    # We need to build the signed adjacency matrix given signs
    
    def build_signed_adj(signs):
        """Build the 2m x 2m signed adjacency matrix."""
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(graph, signs):
        """Check if the lift is connected and 3-regular."""
        from autoresearch_suite.science_tcs_expansion import check_lift
        try:
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, float('inf')
    
    # Initial random signs
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    best_valid = False
    
    # Try multiple random initializations with local search
    n_initializations = 50
    
    for init_idx in range(n_initializations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initial signs
        signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
        
        # Check validity
        valid, radius = is_valid_lift(graph, signs)
        if valid:
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                best_valid = True
        
        # Local search: try flipping each edge
        for flip_iter in range(200):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Flip edge i
                signs[i] = -signs[i]
                
                # Check validity and compute radius
                valid, new_radius = is_valid_lift(graph, signs)
                
                if valid and new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    best_valid = True
                    improved = True
                else:
                    # Revert
                    signs[i] = -signs[i]
            
            if not improved:
                break
    
    # If no valid lift found, try a more systematic approach
    if not best_valid:
        # Try all -1 signs (trivial lift: two disconnected cycles)
        # Actually, all -1 might not be connected. Let's try patterns.
        for pattern_idx in range(20):
            if time.time() - start_time > DEADLINE:
                break
            
            # Create a pattern: alternate signs on cycles
            signs = np.ones(n_edges, dtype=np.float64)
            # Flip some matching edges to create connectivity
            # Matching edges are typically at specific positions
            # Let's try: flip every other matching edge
            # We need to identify which edges are matching edges
            
            # For prism graph, edges are typically ordered as:
            # top cycle, bottom cycle, matching (or some order)
            # Let's just try random patterns with some structure
            
            signs = rng.choice([-1, 1], size=n_edges).astype(np.float64)
            # Ensure at least one matching edge has +1 for connectivity
            # Just check validity
            valid, radius = is_valid_lift(graph, signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                best_valid = True
    
    # Final polish: try to improve best_signs further
    if best_valid and best_signs is not None:
        signs = best_signs.copy()
        for polish_iter in range(100):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE:
                    break
                
                signs[i] = -signs[i]
                valid, new_radius = is_valid_lift(graph, signs)
                
                if valid and new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = -signs[i]
            
            if not improved:
                break
    
    # Return as int array
    if best_signs is None:
        # Fallback: return all +1
        return np.ones(n_edges, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
