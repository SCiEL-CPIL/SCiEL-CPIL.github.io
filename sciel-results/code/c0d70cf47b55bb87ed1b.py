"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy array for fast operations
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: must return before 150s wall-clock kill
    deadline = time.time() + 145.0
    
    # Compute spectral initialization using the Fiedler vector
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Compute eigenvalues and eigenvectors of the Laplacian
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort by eigenvalue (eigh returns in ascending order)
        fiedler = eigvecs[:, 1]
        # Create initial partition based on the median of the Fiedler vector
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(np.int8)
    except Exception:
        # Fallback to alternating initialization
        labels = np.arange(n) % 2
    
    # Local search: 1-flip improvement
    def _cut_size(labels):
        # Count edges crossing the partition
        cross = 0
        for v in range(n):
            cross += adj[v][labels != labels[v]].sum()
        return cross // 2  # Each edge counted twice
    
    def _local_search(labels, max_iters=50):
        """Perform 1-flip local search until no improvement."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            for v in range(n):
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    # Initial local search
    labels = _local_search(labels.copy())
    
    # Iterated Local Search with random perturbations to escape local optima
    best_labels = labels.copy()
    best_cut = _cut_size(best_labels)
    
    while time.time() < deadline - 10.0:
        # Create a perturbed version of the best solution
        perturbed = best_labels.copy()
        
        # Randomly flip a few vertices (2-5 flips)
        num_flips = np.random.randint(2, 6)
        flip_indices = np.random.choice(n, size=num_flips, replace=False)
        perturbed[flip_indices] = 1 - perturbed[flip_indices]
        
        # Run local search on the perturbed solution
        improved_labels = _local_search(perturbed, max_iters=30)
        improved_cut = _cut_size(improved_labels)
        
        # Accept if better
        if improved_cut > best_cut:
            best_labels = improved_labels
            best_cut = improved_cut
    
    return best_labels
# EVOLVE-BLOCK-END
