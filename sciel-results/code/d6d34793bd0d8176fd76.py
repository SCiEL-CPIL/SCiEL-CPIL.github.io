import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Construct Laplacian
    degree = adjacency.sum(axis=1)
    L = np.diag(degree) - adjacency
    
    # Compute eigenvalues and eigenvectors
    # Use eigh for symmetric matrices
    try:
        vals, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return -degree.astype(float)
        
    # Check if graph is likely connected
    # If the second smallest eigenvalue is very close to 0, the graph is disconnected
    # In that case, the Fiedler vector is not well-defined as a unique "periphery" indicator
    if n > 1 and vals[1] < 1e-6:
        # Fallback to degree heuristic for disconnected graphs
        # Or we could try to process components, but that's complex.
        # Degree is a safe bet.
        return -degree.astype(float)
    
    # Get the Fiedler vector (second smallest eigenvalue)
    fiedler_vec = vecs[:, 1]
    
    # The magnitude of the Fiedler vector is used as the priority score.
    # Higher magnitude = more peripheral = better candidate for independent set.
    scores = np.abs(fiedler_vec)
    
    return scores
