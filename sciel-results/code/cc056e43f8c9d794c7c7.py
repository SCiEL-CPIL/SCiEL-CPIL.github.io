import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    # Convert to float for linear algebra
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Compute degrees
    degrees = A.sum(axis=1)
    
    # Fallback: if graph is empty or time is tight
    if n == 0:
        return np.array([])
        
    # Compute Laplacian L = D - A
    # D is diagonal matrix of degrees
    D = np.diag(degrees)
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    # eigh is for symmetric/Hermitian matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to min-degree if eigen decomposition fails
        return -degrees
    
    # Find the smallest eigenvalue strictly greater than a small tolerance
    # This corresponds to the Fiedler eigenvalue for connected graphs
    # For disconnected graphs, it's the smallest non-zero eigenvalue among components
    tol = 1e-8
    # eigenvalues are sorted in ascending order by eigh
    # Find first index where eigenvalue > tol
    mask = eigenvalues > tol
    if not np.any(mask):
        # All eigenvalues are ~0, graph has no edges (or is a set of isolated nodes)
        return -degrees
    
    idx = np.argmax(mask)
    fiedler_vec = eigenvectors[:, idx]
    
    # Score: High absolute Fiedler value is good (peripheral)
    # Subtract degree to break ties and penalize high-degree nodes slightly
    # The weight 1.0 balances spectral position and degree
    score = np.abs(fiedler_vec) - degrees
    
    # Check time
    if time.time() - start_time > 100:
        # Return best result found so far (which is just the score)
        pass
        
    return score
