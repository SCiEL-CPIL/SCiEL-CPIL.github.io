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
    
    # Compute additional features:
    # 1. Second-order: sum of neighbor's neighbor degrees (2-hop density)
    # 2. Clustering coefficient proxy: fraction of edges among neighbors
    
    # 2-hop: for each vertex, sum of degrees of neighbors' neighbors (excluding self)
    # This captures whether a vertex sits at a "bridge" between dense regions
    two_hop = adj_float @ (adj_float @ deg)
    # Normalize by degree^2 to get mean 2-hop degree
    two_hop_norm = np.zeros(n, dtype=float)
    mask2 = deg > 0
    two_hop_norm[mask2] = two_hop[mask2] / (deg[mask2] ** 2)
    
    # Check deadline
    if time.time() > deadline:
        scores = (-1.0 * deg_norm - 0.3 * k_core_norm + 
                  0.4 * mnd_norm - 0.3 * sid_norm)
        scores[~mask] = 1e6
        return scores
    
    # 3. Number of low-degree neighbors (degree < median)
    median_deg = np.median(deg)
    low_deg_neighbors = (adj_float @ (deg < median_deg).astype(float))
    low_deg_norm = low_deg_neighbors / deg_max
    
    # 4. Eccentricity proxy: use BFS-like approximation via power iteration
    # For N=200, we can do a few iterations of max-plus
    # Actually, let's use a simpler proxy: sum of (1 / (1 + distance))
    # But that's expensive. Instead, use the number of vertices reachable in 2 hops
    two_hop_reach = (adj_float @ (adj_float @ np.ones(n)))
    # Subtract self if connected to self (shouldn't be in simple graph)
    two_hop_reach = np.maximum(two_hop_reach - np.eye(n).sum(axis=1), 0)
    two_hop_reach_norm = two_hop_reach / (n - 1) if n > 1 else np.zeros(n)
    
    # Check deadline
    if time.time() > deadline:
        scores = (-1.0 * deg_norm - 0.3 * k_core_norm + 
                  0.4 * mnd_norm - 0.3 * sid_norm)
        scores[~mask] = 1e6
        return scores
    
    # Normalize new features
    th_max = two_hop_norm.max() if two_hop_norm.max() > 0 else 1.0
    ld_max = low_deg_norm.max() if low_deg_norm.max() > 0 else 1.0
    tr_max = two_hop_reach_norm.max() if two_hop_reach_norm.max() > 0 else 1.0
    
    two_hop_norm = two_hop_norm / th_max
    low_deg_norm = low_deg_norm / ld_max
    two_hop_reach_norm = two_hop_reach_norm / tr_max
    
    # Combined score with refined weights:
    # - Primary: low degree (main driver)
    # - Secondary: low k-core (sparse regions preferred)
    # - Tertiary: high mean neighbor degree (neighbors in dense areas are less likely in MIS)
    # - Quaternary: low sum of inverse neighbor degrees (avoid blocking low-degree candidates)
    # - Quinary: low 2-hop density (avoid vertices in very dense 2-hop neighborhoods)
    # - Senary: low number of low-degree neighbors (don't block many low-degree vertices)
    # - Septenary: low 2-hop reachability (vertices in more connected regions are worse)
    
    w_deg = 1.0       # Weight for degree
    w_kcore = 0.25    # Weight for k-core
    w_mnd = 0.35      # Weight for mean neighbor degree
    w_sid = 0.25      # Weight for sum inverse neighbor degree
    w_2hop = 0.15     # Weight for 2-hop density
    w_ld = 0.10       # Weight for low-degree neighbors
    w_tr = 0.05       # Weight for 2-hop reachability
    
    scores = (-w_deg * deg_norm - w_kcore * k_core_norm + 
              w_mnd * mnd_norm - w_sid * sid_norm
              - w_2hop * two_hop_norm - w_ld * low_deg_norm
              - w_tr * two_hop_reach_norm)
    
    # Isolated vertices (degree 0) should be picked first
    scores[~mask] = 1e6
    
    return scores
# EVOLVE-BLOCK-END
