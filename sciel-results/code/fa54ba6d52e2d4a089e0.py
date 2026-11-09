import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    active = np.ones(n, dtype=bool)
    scores = np.zeros(n, dtype=float)
    
    step = 0
    while active.any():
        # Calculate degrees within the active subgraph
        # adjacency is bool matrix, active is bool vector
        # degrees[i] = sum of adjacency[i][j] for all j where active[j] is True
        # Using matrix-vector multiplication or sum along axis
        degrees = (adjacency * active[:, None]).sum(axis=1)
        
        # Mask out inactive vertices so they don't get selected
        # Set degrees of inactive vertices to infinity so they are never min
        degrees[~active] = np.inf
        
        # Select the vertex with the minimum degree
        # If there are ties, np.argmin picks the first one, which is fine
        v = np.argmin(degrees)
        
        # Assign score: earlier selections get higher scores
        scores[v] = n - step
        
        # Remove v and its neighbors from active
        # Neighbors of v are where adjacency[v] is True
        # We need to deactivate v and all u such that adjacency[v][u] is True
        # AND u is currently active (though deactivating inactive ones is harmless)
        neighbors_to_remove = adjacency[v] | (np.arange(n) == v)
        active[neighbors_to_remove] = False
        
        step += 1
        
    return scores
