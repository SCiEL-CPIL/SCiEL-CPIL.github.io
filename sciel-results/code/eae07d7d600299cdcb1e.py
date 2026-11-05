"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def compute_k_core(adj):
    """Compute k-core numbers for a graph."""
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)
        
    degrees = adj.sum(axis=1).astype(int)
    k_core = np.zeros(n, dtype=int)
    max_deg = np.max(degrees) if n > 0 else 0
    
    degs = degrees.copy()
    removed = np.zeros(n, dtype=bool)
    
    # Peeling process
    # We iterate k from 1 to max_deg + 1
    # If a node has current degree < k, it is not in the k-core.
    # We remove it and assign it k_core = k - 1.
    
    for k in range(1, max_deg + 2):
        # Find any node with degree < k and not removed
        while True:
            # Find candidate with min degree among those < k
            # Since N is small, linear scan is fine
            cand = -1
            min_deg = float('inf')
            for i in range(n):
                if not removed[i] and degs[i] < k:
                    if degs[i] < min_deg:
                        min_deg = degs[i]
                        cand = i
            
            if cand == -1:
                break
            
            # Remove cand
            removed[cand] = True
            k_core[cand] = k - 1
            
            # Update neighbors
            neighbors = np.where(adj[cand] > 0)[0]
            for v in neighbors:
                if not removed[v]:
                    degs[v] -= 1
                    
    return k_core


def priority(adjacency):
    """
    Compute priority scores for vertices.
    Higher score means the vertex is more likely to be selected earlier by the greedy decoder.
    The strategy is to prefer vertices that have a low degree (low cost to pick) 
    but whose neighbors have high degrees (high benefit of removing them),
    and are in sparse regions (low k-core).
    Score = mean_neighbor_degree - degree - 0.5 * k_core
    """
    # Ensure adjacency is a float matrix for matrix multiplication
    adj = np.asarray(adjacency, dtype=float)
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute sum of degrees of neighbors
    neighbor_degree_sums = adj.dot(degrees)
    
    # Compute mean degree of neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = neighbor_degree_sums / degrees
        mean_neighbor_degrees = np.nan_to_num(mean_neighbor_degrees, nan=0.0)
        
    # Compute k-core numbers
    k_core = compute_k_core(adj)
    
    # Score:
    # 1. mean_neighbor_degrees: high is good
    # 2. - degrees: low is good
    # 3. - k_core: low is good (sparse regions)
    
    scores = mean_neighbor_degrees - degrees - 0.5 * k_core
    
    return scores
# EVOLVE-BLOCK-END
