"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solves the Maximum Cut problem using Spectral Initialization and Simulated Annealing.
    
    Args:
    adjacency: A boolean matrix representing the graph adjacency.
    
    Returns:
    A numpy array of 0s and 1s representing the partition labels.
    """
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: Stop 5 seconds before the 150s limit to be safe
    start_time = time.time()
    deadline = start_time + 145.0
    
    if n == 0:
        return np.array([], dtype=np.int8)
    if n == 1:
        return np.array([0], dtype=np.int8)
        
    # 1. Spectral Initialization using the Fiedler Vector
    # Compute Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Compute eigenvalues and eigenvectors
    # For N=100, this is very fast
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort is guaranteed by eigh
        fiedler_vec = eigvecs[:, 1]
    except Exception:
        # Fallback to random initialization if spectral fails
        fiedler_vec = np.random.randn(n)
        
    # Initialize labels based on the sign of the Fiedler vector
    # We map positive to 1, negative to 0
    labels = (fiedler_vec > 0).astype(np.int8)
    
    # Ensure we have both 0s and 1s if possible (though sign usually guarantees this for connected graphs)
    if np.all(labels == 0):
        labels[0] = 1
    elif np.all(labels == 1):
        labels[0] = 0

    # 2. Setup for Simulated Annealing
    # Precompute neighbor lists for faster iteration if needed, 
    # but for N=100, iterating over all nodes or using matrix ops is fine.
    # We maintain n_same and n_diff counts for each vertex to calculate delta in O(1) per flip check.
    
    # n_same[v] = number of neighbors of v with the SAME label as v
    # n_diff[v] = number of neighbors of v with the DIFFERENT label as v
    # Cut size contribution of v is n_diff[v].
    # Total Cut = sum(n_diff[v]) / 2 (since each edge is counted twice, once for each endpoint)
    # Actually, if we just sum n_diff[v], it's 2 * Cut.
    
    n_same = np.zeros(n, dtype=np.int32)
    n_diff = np.zeros(n, dtype=np.int32)
    
    for v in range(n):
        neighbors = np.where(adj[v] == 1)[0]
        if len(neighbors) == 0:
            continue
        # Count how many neighbors have the same label
        same_count = np.sum(labels[neighbors] == labels[v])
        n_same[v] = same_count
        n_diff[v] = len(neighbors) - same_count
        
    current_cut = np.sum(n_diff) / 2.0
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Simulated Annealing Parameters
    # We want to run as many steps as possible before the deadline.
    # Each step is O(degree) which is small.
    T = 10.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.01
    
    # Precompute degrees for random selection weighting? 
    # Uniform random selection of vertex is usually sufficient.
    
    # To speed up, we can vectorize the delta calculation partially, 
    # but the update is sequential.
    
    while time.time() < deadline and T > min_T:
        # Pick a random vertex
        v = np.random.randint(0, n)
        
        # Calculate delta if we flip v
        # If v is in group 0:
        # Current contribution to cut from v's perspective: n_diff[v]
        # New contribution: n_same[v] (because neighbors that were 'same' become 'diff')
        # Delta = New - Old = n_same[v] - n_diff[v]
        
        delta = n_same[v] - n_diff[v]
        
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # Accept with probability exp(delta / T)
            # If delta is very negative, exp is ~0
            if T > 0:
                if np.random.rand() < np.exp(delta / T):
                    accept = True
        
        if accept:
            # Flip v
            old_label = labels[v]
            new_label = 1 - old_label
            labels[v] = new_label
            
            # Update counts for v
            n_same[v], n_diff[v] = n_diff[v], n_same[v]
            
            # Update counts for neighbors of v
            # For each neighbor u of v:
            # If u had same label as old v, it now has diff label.
            # So n_same[u] decreases by 1, n_diff[u] increases by 1.
            # If u had diff label from old v, it now has same label.
            # So n_same[u] increases by 1, n_diff[u] decreases by 1.
            
            neighbors = np.where(adj[v] == 1)[0]
            if len(neighbors) > 0:
                # Neighbors that had the SAME label as the OLD label of v
                # These are neighbors where labels[u] == old_label
                # Note: labels[v] is already updated, so we compare against old_label
                same_old_mask = (labels[neighbors] == old_label)
                
                # For these neighbors, 'same' becomes 'diff'
                n_same[neighbors[same_old_mask]] -= 1
                n_diff[neighbors[same_old_mask]] += 1
                
                # Neighbors that had the DIFFERENT label from the OLD label of v
                diff_old_mask = (labels[neighbors] != old_label)
                
                # For these neighbors, 'diff' becomes 'same'
                n_same[neighbors[diff_old_mask]] += 1
                n_diff[neighbors[diff_old_mask]] -= 1
            
            # Update current cut
            current_cut += delta / 2.0
            
            # Check if best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= cooling_rate
        
    return best_labels
# EVOLVE-BLOCK-END
