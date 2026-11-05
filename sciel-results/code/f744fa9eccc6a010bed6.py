"""Program for Maximum Independent Set via dynamic exclusion process."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign priority scores to vertices based on a simulated parallel discrete-time
    exclusion process. Vertices with fewer active neighbors become eligible earlier.
    The score reflects the earliest time step at which a vertex becomes eligible,
    capturing dynamic interference patterns.
    
    Higher score = selected earlier in greedy decoder.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Work with integer adjacency for efficiency
    adj = adjacency.astype(np.int32)
    
    # Pre-compute static degrees for tie-breaking
    static_degrees = adj.sum(axis=1).astype(np.int32)
    
    # Track active (not yet selected) vertices
    remaining = np.ones(n, dtype=bool)
    
    # For each vertex, track the number of currently active neighbors
    # Initially, active_neighbors[v] = degree[v]
    active_neighbors = static_degrees.copy()
    
    # We'll simulate rounds. In each round, vertices with 0 active neighbors are selected.
    # If none, we select the vertex with minimum active neighbors, breaking ties by static degree.
    
    eligibility_round = np.full(n, -1, dtype=np.int32)
    
    current_round = 0
    while remaining.any():
        # Find vertices that are remaining and have 0 active neighbors
        eligible = remaining & (active_neighbors == 0)
        
        if not eligible.any():
            # No vertices with 0 active neighbors.
            # Break deadlock: select vertex with min active neighbors.
            # Tie-breaker: min static degree, then min index.
            remaining_indices = np.where(remaining)[0]
            min_neighbors = active_neighbors[remaining_indices].min()
            candidates = remaining_indices[active_neighbors[remaining_indices] == min_neighbors]
            
            # Sort candidates by static degree, then by index
            cand_static_degrees = static_degrees[candidates]
            # lexsort sorts by last key first, so (index, static_degree)
            # We want to sort by static_degree primarily, then index.
            # So keys are (index, static_degree) -> sort by static_degree, then index
            # np.lexsort((candidates, cand_static_degrees))
            sort_indices = np.lexsort((candidates, cand_static_degrees))
            chosen = candidates[sort_indices[0]]
            
            eligible = np.zeros(n, dtype=bool)
            eligible[chosen] = True
        
        # Assign eligibility round to all eligible vertices
        eligibility_round[eligible] = current_round
        
        # Remove eligible vertices from remaining and update active_neighbors of their neighbors
        eligible_indices = np.where(eligible)[0]
        remaining[eligible_indices] = False
        
        # For each neighbor of the selected vertices, decrement active_neighbors
        # adj[eligible_indices, :] gives the neighbors of selected vertices
        neighbor_updates = adj[eligible_indices, :].sum(axis=0)
        active_neighbors -= neighbor_updates
        
        current_round += 1
    
    # Now assign scores: higher score = should be selected earlier
    # score = -eligibility_round * n - vertex_index
    # This ensures vertices with lower eligibility_round get higher scores.
    # For ties in eligibility_round, lower vertex_index gets higher score.
    scores = -eligibility_round.astype(float) * n - np.arange(n, dtype=float)
    
    return scores
# EVOLVE-BLOCK-END
