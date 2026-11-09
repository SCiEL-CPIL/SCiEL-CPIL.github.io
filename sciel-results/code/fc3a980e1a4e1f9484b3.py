import numpy as np

def priority(adjacency):
    A = adjacency.astype(float)
    
    # Compute eigenvalues and eigenvectors of the symmetric adjacency matrix
    # eigh is for symmetric/Hermitian matrices, returns sorted eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Get the eigenvector corresponding to the largest eigenvalue
    # The largest eigenvalue is at the end of the sorted array
    principal_eigenvector = eigenvectors[:, -1]
    
    # Ensure the eigenvector is oriented such that its sum is positive
    # This ensures that "central" nodes (large magnitude) have positive values
    if np.sum(principal_eigenvector) < 0:
        principal_eigenvector = -principal_eigenvector
        
    # Score: Negate the eigenvector so that nodes with low centrality 
    # (small values in principal eigenvector) get high scores.
    scores = -principal_eigenvector
    
    return scores
