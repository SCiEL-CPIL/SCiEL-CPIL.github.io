"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Strategy:
    1. Local Heuristic: Prioritize vertices with low degree and high "clustering coefficient" of neighbors.
       - Low degree: Fewer vertices blocked.
       - High neighbor clustering: Neighbors are tightly connected, so blocking them is efficient 
         (they would have blocked each other anyway). This is approximated by the sum of degrees of neighbors.
    2. Spectral Heuristic: Use the Fiedler vector (eigenvector of the second smallest eigenvalue of the Laplacian).
       This identifies the global partition. We prioritize vertices that are "central" in their partition 
       or simply use the sign/magnitude to guide the greedy process towards a coherent independent set.
       
    The score combines local efficiency with global structural awareness.
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    # This term rewards vertices whose neighbors are "busy" (high degree), 
    # implying that blocking them is a high-value move.
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate mean degree of neighbors (clustering proxy)
    # Use a safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.where(degrees > 0, neighbor_degree_sums / degrees, 0.0)
        
    # Base priority: 
    # We want to pick low degree vertices first.
    # But if a low degree vertex is connected to high degree hubs, it's even better.
    # Score = -degree + mean_neighbor_deg
    base_priorities = -degrees + mean_neighbor_deg
    
    # Check time budget
    if time.time() - start_time > 100:
        return base_priorities.astype(float)
        
    # Spectral Heuristic
    try:
        # Compute Laplacian L = D - A
        D = np.diag(degrees)
        L = D - adj.astype(np.float64)
        
        # Compute eigenvalues and eigenvectors
        # We only need the first few eigenvalues/vectors
        # np.linalg.eigh is O(N^3) but for N=200 it's fast.
        # To be safer and faster, we can use scipy.sparse if available, 
        # but let's stick to numpy for simplicity and reliability unless we hit timeout.
        
        if time.time() - start_time > 110:
            return base_priorities.astype(float)
            
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Find the Fiedler vector (index of second smallest eigenvalue)
        # For connected graphs, eigvals[0] is 0. eigvals[1] is the Fiedler value.
        # For disconnected graphs, there are multiple 0s.
        # We want the first non-zero eigenvalue's eigenvector.
        
        non_zero_idx = np.where(eigvals > 1e-8)[0]
        if len(non_zero_idx) > 0:
            fiedler_idx = non_zero_idx[0]
        else:
            # Graph is completely disconnected (no edges)
            # Fiedler vector is not well-defined or trivial.
            # Just return base priorities.
            return base_priorities.astype(float)
            
        fiedler_vec = eigvecs[:, fiedler_idx]
        
        # The Fiedler vector values range roughly [-1, 1] after normalization.
        # We can use the magnitude of the Fiedler vector as a measure of how "central" 
        # a vertex is to the partition. Or we can use the value itself to break ties 
        # in a way that respects the global structure.
        
        # Let's try adding the Fiedler vector directly.
        # Vertices with high Fiedler values are on one side of the cut.
        # Vertices with low Fiedler values are on the other.
        # By adding this, we slightly prioritize one side over the other, 
        # which might help in creating a more independent set if the cut is sparse.
        
        # Normalize Fiedler vector to have zero mean and unit variance for consistency
        fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
        std = np.std(fiedler_vec)
        if std > 1e-8:
            fiedler_vec = fiedler_vec / std
        else:
            fiedler_vec = np.zeros_like(fiedler_vec)
            
        # Weight for spectral term
        # Base priorities are dominated by -degree (range ~ -200 to 0) and mean_neighbor_deg (range ~ 0 to 200).
        # The spectral term should be a small perturbation.
        # Let's use alpha = 2.0 to have a noticeable but not dominant effect.
        alpha = 2.0
        
        priorities = base_priorities + alpha * fiedler_vec
        
    except Exception:
        # If spectral computation fails, fall back to base priorities
        priorities = base_priorities
        
    # Final deadline guard
    if time.time() - start_time > 115:
        return base_priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
