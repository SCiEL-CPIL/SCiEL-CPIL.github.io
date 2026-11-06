"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a score to every vertex based on its core number and degree.
    Vertices with lower core numbers (more peripheral) and lower degrees
    are prioritized. This approach uses the k-core decomposition concept,
    which identifies the structural "shells" of the graph. Picking vertices
    from the outer shells (low core number) tends to yield larger independent
    sets in random graphs like Erdos-Renyi.
    """
    adj = np.asarray(adjacency, dtype=np.int32)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Compute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # If graph is empty (no edges), all degrees are 0.
    # All vertices are independent. Any order works.
    if np.all(degrees == 0):
        return np.zeros(n)
    
    # 2. Compute Core Numbers
    # We use an O(N^2) simulation of k-core decomposition suitable for N=200.
    # 1. Create a set of active vertices.
    # 2. While active vertices exist:
    #    a. Find the minimum degree in the active subgraph. Let this be k.
    #    b. All vertices with degree k are assigned core number k.
    #    c. Remove them from the active set and update degrees of their neighbors.
    
    active = set(range(n))
    curr_deg = degrees.copy()
    core_numbers = np.zeros(n, dtype=np.int32)
    
    # Pre-compute neighbors for speed
    neighbors = [np.where(adj[v] > 0)[0] for v in range(n)]
    
    while active:
        # Find min degree among active vertices
        # Since N is small, iterating through active set is fine
        min_deg = min(curr_deg[v] for v in active)
        
        # Identify all vertices with this min degree
        to_remove = [v for v in active if curr_deg[v] == min_deg]
        
        # Assign core number
        for v in to_remove:
            core_numbers[v] = min_deg
            active.remove(v)
            
        # Update degrees of neighbors
        for v in to_remove:
            for u in neighbors[v]:
                if u in active:
                    curr_deg[u] -= 1
                    
    # 3. Calculate Scores
    # We want low core number and low degree to have HIGH score.
    # Score = - (alpha * core_number + beta * degree)
    
    max_core = np.max(core_numbers)
    max_deg = np.max(degrees)
    
    if max_core > 0:
        norm_core = core_numbers / max_core
    else:
        norm_core = np.zeros(n)
        
    if max_deg > 0:
        norm_deg = degrees / max_deg
    else:
        norm_deg = np.zeros(n)
        
    # Combine: We prioritize low core number heavily.
    # Weight 0.7 for core, 0.3 for degree.
    composite = 0.7 * norm_core + 0.3 * norm_deg
    
    # Higher score is better, so we negate.
    score = -composite
    
    return score
# EVOLVE-BLOCK-END
