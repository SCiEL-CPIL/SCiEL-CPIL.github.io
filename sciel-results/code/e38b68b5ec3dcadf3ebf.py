"""Spectral partitioning with iterated local search refinement for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array if needed
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Build graph Laplacian: L = D - A
    D = np.diag(degrees)
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
        fiedler_vector = eigenvectors[:, 1]
    except np.linalg.LinAlgError:
        # Fallback to random
        fiedler_vector = np.random.randn(n)
    
    # Create initial partition
    labels = np.where(fiedler_vector >= 0, 0, 1).astype(int)
    
    def cut_size(labels):
        # Efficient calculation of cut size
        total = 0
        for i in range(n):
            if labels[i] == 0:
                total += A[i][labels == 1].sum()
            else:
                total += A[i][labels == 0].sum()
        return total / 2.0

    def compute_gain1(labels):
        labels_col = labels[:, None]
        labels_row = labels[None, :]
        is_crossing = (labels_col != labels_row).astype(float)
        C = (A * is_crossing).sum(axis=1)
        return degrees - 2 * C

    def find_best_1flip(gains):
        max_idx = np.argmax(gains)
        max_val = gains[max_idx]
        if max_val <= 1e-9:
            return -1, 0
        return max_idx, max_val

    def find_best_2flip(labels, gains):
        g_u = gains[:, None]
        g_v = gains[None, :]
        gain_matrix = g_u + g_v + 2 * A
        
        mask_u = (labels == 0)[:, None]
        mask_v = (labels == 1)[None, :]
        valid_mask = mask_u & mask_v
        
        gain_matrix[~valid_mask] = -np.inf
        
        max_idx = np.argmax(gain_matrix)
        max_val = gain_matrix.flat[max_idx]
        
        if max_val <= 1e-9:
            return -1, -1, 0
            
        u, v = np.unravel_index(max_idx, gain_matrix.shape)
        return u, v, max_val

    def local_search(labels):
        labels = labels.copy()
        improved = True
        max_iters = 200
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            gains = compute_gain1(labels)
            
            # Try 1-flip
            u, gain_val = find_best_1flip(gains)
            if gain_val > 1e-9:
                labels[u] = 1 - labels[u]
                improved = True
                continue
            
            # Try 2-flip
            u, v, gain_val2 = find_best_2flip(labels, gains)
            if gain_val2 > 1e-9:
                labels[u] = 1 - labels[u]
                labels[v] = 1 - labels[v]
                improved = True
                continue
                
        return labels

    # Initial local search on spectral partition
    current_labels = local_search(labels)
    best_cut = cut_size(current_labels)
    best_labels = current_labels.copy()
    
    # Iterated Local Search
    num_restarts = 100
    for _ in range(num_restarts):
        # Perturb the current best solution
        perturbed_labels = best_labels.copy()
        # Flip 2 to 5 nodes
        k = np.random.randint(2, 6)
        if k > 0:
            flip_indices = np.random.choice(n, k, replace=False)
            perturbed_labels[flip_indices] = 1 - perturbed_labels[flip_indices]
        
        # Run local search on perturbed solution
        refined_labels = local_search(perturbed_labels)
        
        # Check if it's better
        new_cut = cut_size(refined_labels)
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = refined_labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
