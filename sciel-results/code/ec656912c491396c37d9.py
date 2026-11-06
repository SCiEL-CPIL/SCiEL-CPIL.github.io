"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Compute priority scores for vertices using the Fiedler vector (eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian).
    
    The magnitude of the Fiedler vector components identifies vertices at the
    periphery of the graph's main components. These vertices are less central
    to connectivity, making them safer to include in an independent set early
    in the greedy process without fragmenting the remaining subgraph or
    blocking high-value internal nodes.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Ensure adjacency is a float matrix
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Symmetrize just in case, though ER graphs are undirected
    adj = (adj + adj.T) / 2.0
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute Laplacian L = D - A
    D = np.diag(degrees)
    L = D - adj
    
    try:
        # Use sparse eigensolver if available and beneficial, otherwise dense
        # For n=200, dense is very fast and robust.
        # We want the smallest eigenvalues.
        # scipy.sparse.linalg.eigsh is good for sparse, but for n=200, np.linalg.eigh is sufficient.
        
        # Compute eigenvalues and eigenvectors
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigenvalues[0] is 0 (for connected graphs), eigenvalues[1] is the Fiedler value
        if n > 1:
            fiedler_vector = eigenvectors[:, 1]
        else:
            fiedler_vector = np.array([1.0])
            
    except Exception as e:
        # Fallback to degree-based if spectral fails (unlikely for n=200)
        # Low degree is generally good for greedy MIS
        return -degrees

    # The magnitude of the Fiedler vector indicates how "far" a node is from the
    # "center" of the graph's connectivity structure.
    # Higher magnitude -> more peripheral -> safer to pick early.
    
    # We use the absolute value.
    scores = np.abs(fiedler_vector)
    
    # Optional: Normalize to a reasonable range, though greedy only cares about order.
    # Subtracting a small amount of degree might help break ties or refine, 
    # but let's stick to the pure spectral signal first as requested by the strategy.
    # However, pure Fiedler vector magnitude can be zero for symmetric nodes.
    # Let's check if we need a tie-breaker.
    # If scores are identical, the decoder's behavior depends on stable sort or random.
    # To be safe, we can subtract a tiny fraction of degree to prefer lower degree on ties.
    # This maintains the spectral primary signal but adds a local heuristic secondary signal.
    
    # Normalize scores to [0, 1] to mix with degree if needed?
    # Let's try just the spectral score first.
    
    return scores
# EVOLVE-BLOCK-END
