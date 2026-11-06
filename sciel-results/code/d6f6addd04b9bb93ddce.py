"""Rewritten program using a Degree-Primary Spectral approach for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assign priority scores to vertices.
    Strategy: Primary priority is given to low-degree vertices (standard MIS heuristic).
    Secondary priority (tie-breaking/modulation) is based on the Fiedler vector magnitude.
    Vertices with higher absolute Fiedler component (more peripheral) are given a slight boost,
    hypothesizing that peripheral nodes are less likely to cause fragmentation or conflict with 
    high-value core nodes.
    
    Returns an array of scores where higher score = higher priority (selected first).
    """
    n = adjacency.shape[0]
    
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    adjacency = adjacency.astype(float)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute degree vector
    degrees = adjacency.sum(axis=1)
    
    # Base score: Inverse degree (lower degree -> higher score)
    # We use -degree so that lower degree vertices have higher (less negative) scores.
    # To avoid division by zero or scaling issues, we just use -degree.
    scores = -degrees.copy()
    
    # Spectral enhancement:
    # Compute the Fiedler vector to identify peripheral vs central nodes.
    if n > 2:
        try:
            A_sparse = csr_matrix(adjacency)
            # Build Laplacian L = D - A
            D_sparse = csr_matrix(np.diag(degrees))
            L = D_sparse - A_sparse
            
            # Find the 2 smallest eigenvalues/eigenvectors
            # sigma=0 with which='LM' finds eigenvalues closest to 0
            # For Laplacian, these are lambda_1=0 and lambda_2 (Fiedler)
            eigenvalues, eigenvectors = eigsh(L, k=2, sigma=0, which='LM')
            
            # Ensure we pick the correct eigenvector (the one for lambda_2, not lambda_1)
            # lambda_1 should be ~0. lambda_2 > 0.
            # Sort eigenvalues
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # The Fiedler vector is the eigenvector for the second smallest eigenvalue (index 1)
            # Note: If the graph is disconnected, lambda_2 will be 0 and there will be multiple 
            # zero eigenvalues. In that case, the "Fiedler" vector is not unique. 
            # For ER graphs with n=200, they are likely connected.
            fiedler_vec = eigenvectors[:, 1]
            
            # Normalize the Fiedler vector to [0, 1] range for consistent scaling
            f_min = np.min(np.abs(fiedler_vec))
            f_max = np.max(np.abs(fiedler_vec))
            if f_max > f_min:
                f_norm = (np.abs(fiedler_vec) - f_min) / (f_max - f_min)
            else:
                f_norm = np.zeros(n)
            
            # Add a small spectral boost to peripheral nodes (high f_norm)
            # This acts as a sophisticated tie-breaker for nodes with similar degrees
            boost_weight = 0.5
            scores += boost_weight * f_norm
            
        except Exception:
            # Fallback: if spectral computation fails, rely purely on degree
            pass
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
