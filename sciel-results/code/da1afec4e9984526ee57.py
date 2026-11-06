"""Spectral partitioning with iterated local search refinement for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array if needed
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Precompute neighbor structure for faster local search
    # For each vertex, store its neighbors as a list for quick access
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]
    
    def cut_size(labels):
        """Compute cut size efficiently."""
        total = 0
        for i in range(n):
            if labels[i] == 0:
                total += A[i, labels == 1].sum()
            else:
                total += A[i, labels == 0].sum()
        return total / 2.0

    def local_search(labels):
        """Perform 1-flip local search until convergence with precomputed neighbors."""
        improved = True
        indices = np.arange(n)
        
        while improved:
            improved = False
            np.random.shuffle(indices)
            for v in indices:
                current_label = labels[v]
                other_label = 1 - current_label
                nbrs = neighbors[v]
                if len(nbrs) == 0:
                    continue
                # Count edges to same and other side using precomputed neighbors
                edges_same = labels[nbrs][labels[nbrs] == current_label].sum()
                edges_crossing = labels[nbrs][labels[nbrs] == other_label].sum()
                
                if edges_same > edges_crossing:
                    labels[v] = other_label
                    improved = True
        
        return labels

    # Compute eigenvalues and eigenvectors of graph Laplacian
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple alternating pattern
        return np.arange(n) % 2
    
    # Generate multiple initial partitions from different eigenvectors
    initial_labels_list = []
    
    # Fiedler vector (index 1) - most important
    fiedler_vector = eigenvectors[:, 1]
    labels_fiedler = np.where(fiedler_vector >= 0, 0, 1).astype(int)
    initial_labels_list.append(labels_fiedler)
    
    # Try a few other low-frequency eigenvectors for diversity
    for idx in [2, 3, 4]:
        if idx < n:
            vec = eigenvectors[:, idx]
            labels_vec = np.where(vec >= 0, 0, 1).astype(int)
            initial_labels_list.append(labels_vec)
    
    # Also try negated Fiedler (should be equivalent but ensures both orientations)
    labels_fiedler_neg = np.where(fiedler_vector <= 0, 0, 1).astype(int)
    initial_labels_list.append(labels_fiedler_neg)
    
    # Run local search on each initial partition and keep the best
    best_labels = None
    best_cut = -1
    
    for init_labels in initial_labels_list:
        refined = local_search(init_labels.copy())
        cut_val = cut_size(refined)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined.copy()
    
    # Iterated Local Search with more restarts and adaptive perturbation
    num_restarts = 50
    for r in range(num_restarts):
        # Perturb the current best solution
        perturbed_labels = best_labels.copy()
        
        # Adaptive perturbation: start with smaller perturbations, increase over time
        base_k = 1 + int(r / 10)
        k = np.random.randint(max(1, base_k), max(2, base_k + 3))
        k = min(k, n - 1)
        
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
