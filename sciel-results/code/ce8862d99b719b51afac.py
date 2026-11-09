"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Heuristic:
    1. Primary: Prefer vertices with lower degree.
    2. Secondary: Prefer vertices whose neighbors have higher average degrees.
       (Blocking hubs is efficient).
    3. Tertiary: Prefer vertices that are part of many triangles.
       (Blocking dense clusters is efficient).
    4. Quaternary: Spectral term (Fiedler vector) for global structure.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate mean degree of neighbors
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1.0)
    
    # Base priority
    base_priorities = -degrees + mean_neighbor_deg
    
    # Calculate Triangle Count
    # A^2 counts paths of length 2.
    # (A * A^2)_{vu} is non-zero only if A_{vu}=1, in which case it is the number of common neighbors.
    # Summing over u gives twice the number of triangles containing v.
    A2 = adj @ adj
    tri_sum = (adj * A2).sum(axis=1).astype(np.float64)
    tri_count = 0.5 * tri_sum
    
    # Add triangle count to score
    # Scale it down as it might be large
    priorities = base_priorities + 0.5 * tri_count
    
    # Deadline guard
    if time.time() - start_time > 100:
        return priorities.astype(float)
        
    # Spectral Heuristic
    try:
        D = np.diag(degrees)
        L = D - adj.astype(np.float64)
        
        if time.time() - start_time > 110:
            return priorities.astype(float)
            
        eigvals, eigvecs = np.linalg.eigh(L)
        
        non_zero_idx = np.where(eigvals > 1e-8)[0]
        if len(non_zero_idx) > 0:
            fiedler_idx = non_zero_idx[0]
        else:
            fiedler_idx = 1
            
        fiedler_vec = eigvecs[:, fiedler_idx]
        fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
        std = np.std(fiedler_vec)
        if std > 1e-8:
            fiedler_vec = fiedler_vec / std
        else:
            fiedler_vec = np.zeros_like(fiedler_vec)
            
        # Add spectral term
        priorities += 0.5 * fiedler_vec
        
    except Exception:
        pass
        
    # Final deadline guard
    if time.time() - start_time > 115:
        return base_priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
