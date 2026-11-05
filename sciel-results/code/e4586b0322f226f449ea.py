"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Compute priority scores for vertices using a spectral approach.
    We compute the Fiedler vector (eigenvector of the Laplacian corresponding to the second smallest eigenvalue).
    Vertices with large magnitude in the Fiedler vector are considered 'peripheral' or 'boundary' vertices 
    that separate the graph. In the context of the greedy decoder for Independent Set, picking vertices 
    that are structurally distinct (high Fiedler magnitude) might help in maintaining a larger independent set 
    by effectively partitioning the graph into independent regions.
    
    However, the standard heuristic for MIS is often to pick low-degree nodes. 
    Let's combine: 
    1. Fiedler magnitude (structural importance/separability)
    2. Low degree (greedy preference)
    
    The prompt specifically asks to use the Fiedler vector magnitude as the priority score.
    "using its magnitude as the priority score"
    
    So, Score = |Fiedler Vector Component|
    
    But wait, if we just use magnitude, we might pick high-degree nodes if they happen to be at the boundary.
    The directive says: "Use a spectral approach ... and using its magnitude as the priority score".
    
    Let's implement exactly that. 
    Note: The Laplacian L = D - A.
    Smallest eigenvalue is 0 (vector of ones).
    Second smallest is Fiedler value.
    
    We need to be careful with disconnected graphs. Erdos-Renyi with 200 nodes and p~0.5 (typical for these tasks) is likely connected.
    If disconnected, the multiplicity of 0 eigenvalue > 1. We should handle this or assume connected.
    For robustness, if the graph is disconnected, the "Fiedler vector" isn't unique. 
    However, for ER(200, p), it's almost surely connected if p > log(n)/n.
    
    Let's compute the eigenvectors of the Laplacian.
    To save time and avoid full eigendecomposition, we could use scipy.sparse.linalg.eigsh.
    But standard library + numpy/scipy is allowed.
    
    Time limit: 120s.
    200x200 matrix eigendecomposition is very fast (< 0.1s).
    
    Strategy:
    1. Convert adjacency to float.
    2. Compute Laplacian L = D - A.
    3. Compute eigenvalues and eigenvectors.
    4. Sort eigenvalues.
    5. Take the eigenvector corresponding to the second smallest eigenvalue.
    6. Return the absolute values of this vector as the scores.
    
    Wait, the greedy decoder picks vertices in *descending* score order.
    If we want to pick "peripheral" vertices, and high magnitude indicates periphery, then high score = good.
    
    Let's double check the implication.
    If the graph is a path, the Fiedler vector is a sine wave. The ends have the largest magnitude.
    Picking the ends first for MIS is a good strategy (greedy from ends).
    So high magnitude -> high priority seems correct.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute Degrees
    degrees = adj.sum(axis=1)
    
    # Compute Laplacian L = D - A
    # D is diagonal matrix of degrees
    D = np.diag(degrees)
    L = D - adj
    
    # Compute eigenvalues and eigenvectors
    # np.linalg.eigh is for symmetric matrices, which L is.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple degree-based if eig fails (unlikely)
        return -degrees # Lower degree is better, so negative degree makes high score for low degree
    
    # Sort eigenvalues and their corresponding eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # The smallest eigenvalue should be close to 0.
    # The Fiedler vector is the eigenvector for the second smallest eigenvalue (index 1).
    # If the graph is disconnected, there will be multiple 0 eigenvalues.
    # In that case, the "second smallest" might still be 0.
    # For ER graphs, we assume connectivity. If not, any basis vector for the null space is valid.
    # Let's just pick index 1 as per standard definition.
    
    fiedler_vector = eigenvectors[:, 1]
    
    # The magnitude of the Fiedler vector components
    scores = np.abs(fiedler_vector)
    
    # Ensure we don't have NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 110:
        # This shouldn't happen for n=200, but just in case
        pass
        
    return scores

# EVOLVE-BLOCK-END
