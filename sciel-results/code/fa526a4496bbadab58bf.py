"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute adjacency lists for faster matrix construction if needed, 
    # but for n<=18, direct construction is fine.
    
    def get_radius_and_validity(signs):
        """
        Computes spectral radius and validity.
        Returns (radius, is_valid).
        """
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        
        # Check validity: The lift is connected and 3-regular.
        # For a prism graph, the lift is connected iff the signed graph is "balanced" in a specific way?
        # Actually, check_lift handles this. We must call it.
        try:
            radius, details = check_lift(graph, signs)
            return radius, True
        except Exception:
            return float('inf'), False

    # 1. Generate a diverse set of initial candidates
    candidates = []
    
    # All -1
    candidates.append(np.ones(num_edges, dtype=int) * -1)
    # All +1
    candidates.append(np.ones(num_edges, dtype=int))
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        candidates.append(rng.choice([-1, 1], size=num_edges))
        
    # Structured: Alternating on cycles
    for s_top in [-1, 1]:
        for s_bot in [-1, 1]:
            for s_match in [-1, 1]:
                signs = np.ones(num_edges, dtype=int)
                for i, (u, v) in enumerate(edges):
                    if abs(u - v) == m:
                        signs[i] = s_match
                    elif u < m and v < m:
                        # Top cycle: edge (i, i+1) or (0, m-1)
                        # Determine position
                        if u == 0 and v == m-1:
                            pos = m-1
                        elif v == 0 and u == m-1:
                            pos = m-1
                        else:
                            pos = min(u, v)
                        signs[i] = s_top * ((-1) ** pos)
                    else:
                        # Bottom cycle
                        u_b = u - m
                        v_b = v - m
                        if u_b == 0 and v_b == m-1:
                            pos = m-1
                        elif v_b == 0 and u_b == m-1:
                            pos = m-1
                        else:
                            pos = min(u_b, v_b)
                        signs[i] = s_bot * ((-1) ** pos)
                candidates.append(signs)

    # Find best initial candidate
    best_signs = None
    best_radius = float('inf')
    
    for signs in candidates:
        radius, valid = get_radius_and_validity(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback to a known valid one if any exist, otherwise all -1 (might be invalid but we try)
        best_signs = np.ones(num_edges, dtype=int) * -1
        best_radius, _ = get_radius_and_validity(best_signs)

    # 2. Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995 # Cooling rate
    max_iters = 2000
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    current_valid = (current_radius != float('inf'))
    
    T = T_start
    
    for _ in range(max_iters):
        # Generate neighbor: flip a random edge
        neighbor_signs = current_signs.copy()
        idx = rng.integers(num_edges)
        neighbor_signs[idx] *= -1
        
        neighbor_radius, neighbor_valid = get_radius_and_validity(neighbor_signs)
        
        if not neighbor_valid:
            # Invalid lifts are heavily penalized
            neighbor_radius = float('inf')
            
        if neighbor_radius < current_radius:
            current_signs = neighbor_signs
            current_radius = neighbor_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        else:
            # Accept with probability exp(-(delta)/T)
            delta = neighbor_radius - current_radius
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_signs = neighbor_signs
                    current_radius = neighbor_radius
            else:
                # Should not happen if logic above is correct, but just in case
                current_signs = neighbor_signs
                current_radius = neighbor_radius
        
        T *= alpha
        if T < T_end:
            T = T_end

    return best_signs
# EVOLVE-BLOCK-END
