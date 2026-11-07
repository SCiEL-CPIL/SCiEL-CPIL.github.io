"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2  # prism has 2m vertices
    
    # Identify edge types: top cycle, bottom cycle, matching
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(idx)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Build signed adjacency matrix skeleton for fast updates
    # Pre-store edge endpoints for fast matrix construction
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    def build_matrix(signs):
        """Build the signed adjacency matrix from signs."""
        S = np.zeros((n, n), dtype=np.float64)
        for i in range(num_edges):
            s = signs[i]
            u = edge_u[i]
            v = edge_v[i]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(eigvals)))
    
    def is_connected_fast(signs):
        """Check connectivity using sign product criterion.
        The 2-lift of a prism is connected iff product of signs around
        top cycle == -1 OR product around bottom cycle == -1.
        """
        top_prod = 1
        for idx in top_cycle_edges:
            top_prod *= signs[idx]
        
        if top_prod == -1:
            return True
        
        bottom_prod = 1
        for idx in bottom_cycle_edges:
            bottom_prod *= signs[idx]
        
        return bottom_prod == -1
    
    # Initial random search
    rng = np.random.default_rng(3)
    best_signs = None
    best_radius = np.inf
    
    # Sample random sign vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        if not is_connected_fast(signs):
            continue
        
        radius = compute_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # If no valid found, try structured patterns
    if best_signs is None:
        # Try: all +1 except one -1 in top cycle
        for idx in top_cycle_edges:
            signs = np.ones(num_edges, dtype=np.int8)
            signs[idx] = -1
            if is_connected_fast(signs):
                radius = compute_radius(signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
        
        # Try: all +1 except one -1 in bottom cycle
        for idx in bottom_cycle_edges:
            signs = np.ones(num_edges, dtype=np.int8)
            signs[idx] = -1
            if is_connected_fast(signs):
                radius = compute_radius(signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    if best_signs is None:
        return np.ones(num_edges, dtype=np.int8)
    
    # Local search with 1-flip and 2-flip moves
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Pre-compute which edges are in each category for fast flip evaluation
    top_edge_set = set(top_cycle_edges)
    bottom_edge_set = set(bottom_cycle_edges)
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # 1-flip moves
        for i in range(num_edges):
            if time.time() >= deadline - 2.0:
                break
            
            test_signs = current_signs.copy()
            test_signs[i] = -test_signs[i]
            
            if not is_connected_fast(test_signs):
                continue
            
            radius = compute_radius(test_signs)
            if radius < current_radius - 1e-10:
                current_signs = test_signs
                current_radius = radius
                best_signs = test_signs.copy()
                best_radius = radius
                improved = True
        
        if not improved:
            # 2-flip moves
            two_flip_improved = False
            for i in range(num_edges):
                if time.time() >= deadline - 2.0:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() >= deadline - 2.0:
                        break
                    
                    test_signs = current_signs.copy()
                    test_signs[i] = -test_signs[i]
                    test_signs[j] = -test_signs[j]
                    
                    if not is_connected_fast(test_signs):
                        continue
                    
                    radius = compute_radius(test_signs)
                    if radius < current_radius - 1e-10:
                        current_signs = test_signs
                        current_radius = radius
                        best_signs = test_signs.copy()
                        best_radius = radius
                        two_flip_improved = True
                        break
                if two_flip_improved:
                    break
            
            if not two_flip_improved:
                # Random restarts with structure-aware sampling
                found_improvement = False
                for _ in range(200):
                    if time.time() >= deadline - 2.0:
                        break
                    
                    # Sample with bias toward valid configurations
                    # Ensure at least one -1 in top or bottom cycle
                    signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                    
                    # If not connected, force a -1 in a cycle
                    if not is_connected_fast(signs):
                        # Pick a random edge in top cycle and set to -1
                        if top_cycle_edges:
                            idx = top_cycle_edges[rng.integers(len(top_cycle_edges))]
                            signs[idx] = -1
                    
                    if not is_connected_fast(signs):
                        continue
                    
                    radius = compute_radius(signs)
                    if radius < current_radius - 1e-10:
                        current_signs = signs
                        current_radius = radius
                        best_signs = signs.copy()
                        best_radius = radius
                        found_improvement = True
                        break
                
                if not found_improvement:
                    break
    
    return best_signs
# EVOLVE-BLOCK-END
