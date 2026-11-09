"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices to maximize the size of the
    independent set found by a greedy decoder (descending score, skip neighbors).
    
    Strategy:
    Combine 'effective degree' (degree minus neighborhood density) with core numbers.
    We want to select vertices that are 'cheap' to include (low effective degree)
    and 'peripheral' (low core number).
    
    Score = -(effective_degree * 100 + core_number)
    
    Higher score means we pick it earlier.
    """
    start_time = time.time()
    
    # Convert to numpy array
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Ensure it's a valid square matrix
    if A.shape[0] != A.shape[1]:
        return np.zeros(n)
        
    # 1. Compute degrees
    deg = A.sum(axis=1).astype(int)
    
    # 2. Compute Core Numbers (Peeling Algorithm)
    # This is O(N^2) which is fine for N=200
    degs = deg.copy()
    core = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    
    # Pre-compute neighbor lists for faster access during peeling
    neighbors = [np.where(A[i, :])[0] for i in range(n)]
    
    for _ in range(n):
        # Find active vertex with min degree
        min_deg = np.inf
        v = -1
        for u in range(n):
            if active[u] and degs[u] < min_deg:
                min_deg = degs[u]
                v = u
            
        if v == -1:
            break
            
        core[v] = min_deg
        
        # Remove v
        active[v] = False
        for u in neighbors[v]:
            if active[u]:
                degs[u] -= 1
                
    # 3. Compute Effective Degree
    # effective_degree(v) = degree(v) - edges_in_N(v)
    # edges_in_N(v) = sum_{u in N(v)} sum_{w in N(v), w != u} A[u, w] / 2
    # We can compute this as: for each v, look at submatrix A[N(v), N(v)].
    # Sum of elements in that submatrix is 2 * edges_in_N(v).
    
    effective_deg = np.zeros(n, dtype=float)
    
    for v in range(n):
        # Safety check for time
        if time.time() - start_time > 110.0:
            # If we run out of time, fallback to just using degree and core
            # But the loop is fast, so this is unlikely.
            break
            
        nbrs = neighbors[v]
        d_v = len(nbrs)
        
        if d_v > 1:
            # Extract induced subgraph adjacency
            # Note: A is boolean, so sum is count of edges
            sub_adj = A[np.ix_(nbrs, nbrs)]
            # Sum of sub_adj counts each edge twice (u->w and w->u)
            # Diagonal is 0 (assuming no self loops)
            edges_in_nbr = sub_adj.sum() / 2.0
        else:
            edges_in_nbr = 0.0
            
        effective_deg[v] = deg[v] - edges_in_nbr
        
    # 4. Combine Scores
    # We want to minimize effective_deg and core.
    # So we return negative values.
    # We weight effective_deg more heavily because it captures local density better.
    # Core number captures global structural position.
    
    # Normalize or just use raw values?
    # Raw values: effective_deg is roughly in [0, N], core is in [0, N].
    # Let's use a weighted sum.
    
    scores = -(effective_deg * 10.0 + core)
    
    return scores.astype(float)

# EVOLVE-BLOCK-END
