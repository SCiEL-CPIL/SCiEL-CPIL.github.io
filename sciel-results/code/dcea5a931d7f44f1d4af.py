"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Number of simulation runs
    K = 50 # 50 is usually enough for a robust estimate and keeps runtime low
    
    counts = np.zeros(n)
    
    for _ in range(K):
        # Random priorities
        p = np.random.rand(n)
        
        # Available vertices mask
        available = np.ones(n, dtype=bool)
        
        # We need to find the max priority among available vertices
        # and remove it + neighbors.
        
        while np.any(available):
            # Mask unavailable vertices with -inf
            p_masked = np.where(available, p, -np.inf)
            v = np.argmax(p_masked)
            
            # If no available vertex found (shouldn't happen if available has True)
            if not available[v]:
                break
            
            # Select v
            counts[v] += 1
            
            # Remove v and its neighbors
            # adj[v] is the row for v's neighbors
            # We need to set available[v] and available[neighbors] to False
            # adj[v] is bool array
            # v itself is not in adj[v] (assuming no self-loops, ER graphs don't have them)
            # But to be safe, we set v explicitly.
            
            # Efficient way:
            # available[v] = False
            # available[adj[v]] = False
            
            # This is two vector operations.
            available[v] = False
            if n > 0:
                available[adj[v]] = False
            
    return counts
# EVOLVE-BLOCK-END
