"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    adj_float = adjacency.astype(float)
    deg = adjacency.sum(axis=1).astype(float)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: just negative degree
        scores = -deg.copy()
        mask = deg > 0
        scores[~mask] = 1e6
        return scores
    
    # Compute sum of neighbor degrees for each vertex
    sum_neighbor_deg = adj_float @ deg
    
    # Mean neighbor degree, with safe division
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]
    
    # Sum of inverse neighbor degrees
    # Vertices connected to low-degree vertices are penalized more
    # because blocking a low-degree vertex is "costlier"
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = adj_float @ inv_deg
    
    # Check deadline
    if time.time() > deadline:
        scores = -deg + 0.5 * mean_neighbor_deg - 0.5 * sum_inv_neighbor_deg
        scores[~mask] = 1e6
        return scores
    
    # Compute k-core numbers for structural awareness
    # Lower k-core means vertex is in a sparser region
    k_core = np.zeros(n, dtype=np.int32)
    
    if n > 0:
        # Use a simplified peeling approach for k-core
        active = np.ones(n, dtype=bool)
        cur_deg = deg.copy()
        max_k = int(deg.max()) if n > 0 else 0
        
        # For efficiency with N=200, use iterative peeling
        # We'll compute k-core by removing vertices with degree < k iteratively
        for k in range(1, max_k + 1):
            if time.time() > deadline:
                break
            changed = True
            while changed:
                changed = False
                # Find vertices to remove (degree < k and still active)
                to_remove = np.where(active & (cur_deg < k))[0]
                if len(to_remove) == 0:
                    break
                for v in to_remove:
                    if active[v]:
                        k_core[v] = k - 1
                        active[v] = False
                        changed = True
                        # Update neighbors
                        neighbors = np.where(adjacency[v])[0]
                        for u in neighbors:
                            if active[u]:
                                cur_deg[u] -= 1
        
        # Remaining active vertices get the highest k
        remaining = np.where(active)[0]
        if len(remaining) > 0:
            k_core[remaining] = max_k
    
    # Check deadline
    if time.time() > deadline:
        scores = -deg + 0.5 * mean_neighbor_deg - 0.5 * sum_inv_neighbor_deg
        scores[~mask] = 1e6
        return scores
    
    # Normalize components for stable weighting
    k_core_max = k_core.max() if k_core.max() > 0 else 1.0
    deg_max = deg.max() if deg.max() > 0 else 1.0
    mnd_max = mean_neighbor_deg.max() if mean_neighbor_deg.max() > 0 else 1.0
    sid_max = sum_inv_neighbor_deg.max() if sum_inv_neighbor_deg.max() > 0 else 1.0
    
    k_core_norm = k_core.astype(float) / k_core_max
    deg_norm = deg / deg_max
    mnd_norm = mean_neighbor_deg / mnd_max
    sid_norm = sum_inv_neighbor_deg / sid_max
    
    # Combined score:
    # - Primary: low degree (main driver)
    # - Secondary: low k-core (sparse regions preferred)
    # - Tertiary: high mean neighbor degree (neighbors in dense areas are less likely in MIS)
    # - Quaternary: low sum of inverse neighbor degrees (avoid blocking low-degree candidates)
    
    w_deg = 1.0       # Weight for degree
    w_kcore = 0.3     # Weight for k-core
    w_mnd = 0.4       # Weight for mean neighbor degree
    w_sid = 0.3       # Weight for sum inverse neighbor degree
    
    scores = (-w_deg * deg_norm - w_kcore * k_core_norm + 
              w_mnd * mnd_norm - w_sid * sid_norm)
    
    # Isolated vertices (degree 0) should be picked first
    scores[~mask] = 1e6
    
    return scores
# EVOLVE-BLOCK-END
