import numpy as np

def priority(adjacency):
    """
    Calculate priority scores using the Fiedler vector (spectral partitioning).
    Vertices with high absolute Fiedler values are on the 'periphery' of the 
    graph's main cut, making them good candidates for an independent set as 
    they are less likely to be connected to other selected vertices from the 
    opposite side of the cut.
    
    Fallback to a local degree-based heuristic if the graph is disconnected 
    or eigendecomposition fails.
    """
    A = np.asarray(adjacency, dtype=float)
    
    # Ensure symmetry
    A = (A + A.T) / 2.0
    
    degrees = A.sum(axis=1)
    
    # Fallback heuristic: Score = sum(degrees of neighbors) - degree
    # This favors leaves of hubs.
    fallback_scores = A.dot(degrees) - degrees
    
    try:
        # Compute Laplacian L = D - A
        D = np.diag(degrees)
        L = D - A
        
        # Compute eigenvalues and eigenvectors
        # np.linalg.eigh is efficient for symmetric matrices
        vals, vecs = np.linalg.eigh(L)
        
        # Check for disconnectedness
        # If the second smallest eigenvalue is very close to 0, the graph is disconnected.
        # In this case, the Fiedler vector (index 1) is not well-defined for partitioning 
        # the whole graph into two parts in a useful way for a single IS score.
        if vals[1] < 1e-8:
            return fallback_scores
        
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler_vec = vecs[:, 1]
        
        # Prioritize vertices with largest absolute Fiedler values
        scores = np.abs(fiedler_vec)
        
        return scores
    
    except Exception:
        return fallback_scores
