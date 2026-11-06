"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import laplacian
import random


def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Initialization and Greedy Local Search.
    
    This approach uses the Fiedler vector of the graph Laplacian to create a
    high-quality initial partition, then refines it with a fast local search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # Build Laplacian L = D - A
    # For efficiency, we can compute eigenvectors directly on the matrix
    # L = D - A
    L = np.diag(degrees) - adj
    
    # Compute eigenvectors
    # We need the Fiedler vector (2nd smallest eigenvalue)
    # Using numpy.linalg.eigh for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The eigenvalues are sorted in ascending order
        # The Fiedler vector is the eigenvector corresponding to eigenvalues[1]
        fiedler_vec = eigenvectors[:, 1]
    except Exception:
        # Fallback to random if eigendecomposition fails
        fiedler_vec = np.random.randn(n)

    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        adj_labels = adj @ labels
        crossing = labels * degrees + adj_labels - 2 * labels * adj_labels
        return int(np.sum(crossing) / 2)
    
    def local_search(labels, max_iters=10000):
        """
        Fast Greedy Local Search.
        Optimized by pre-computing adj @ labels and updating it incrementally.
        """
        labels = labels.copy().astype(np.int8)
        current_cut = cut_size(labels)
        
        # Precompute adj @ labels
        adj_labels = adj @ labels
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # We can vectorize the gain calculation for all vertices
            # delta[v] = (1 - 2*labels[v]) * (degrees[v] - 2*adj_labels[v])
            # If delta[v] > 0, flipping v improves the cut.
            
            # However, flipping one vertex changes adj_labels for its neighbors.
            # So we must do this sequentially or in batches carefully.
            # For n=100, sequential is fine.
            
            # To speed up, we can just pick the vertex with the largest positive gain
            # and flip it. This is "Best Improvement" local search.
            # It might be slower per iteration but converges faster.
            
            # Let's stick to "First Improvement" but with a random order to avoid bias?
            # Or just standard sequential.
            
            for v in range(n):
                l = labels[v]
                s = adj_labels[v]
                d = degrees[v]
                
                # Calculate gain for flipping v
                # If l=0, gain = d - 2s. If l=1, gain = 2s - d.
                # gain = (1 - 2*l) * (d - 2*s)
                gain = (1 - 2 * l) * (d - 2 * s)
                
                if gain > 0:
                    # Flip v
                    labels[v] = 1 - l
                    current_cut += gain
                    improved = True
                    
                    # Update adj_labels for neighbors of v
                    # For each neighbor u of v, adj_labels[u] changes by (1 - 2*l_new) - (1 - 2*l_old)?
                    # Actually, adj_labels[u] = sum_{w~u} labels[w].
                    # When we flip v, for any neighbor u of v, labels[v] changes.
                    # Old contribution of v to adj_labels[u] was l.
                    # New contribution is 1-l.
                    # Change in adj_labels[u] = (1-l) - l = 1 - 2l.
                    # Note: l here is the OLD label.
                    # So delta_adj_labels = 1 - 2*l_old = 1 - 2*(1 - labels[v])?
                    # No, labels[v] is now the NEW label.
                    # Let new_l = 1 - l.
                    # Change = new_l - l = (1-l) - l = 1 - 2l.
                    
                    # We need to update adj_labels for all neighbors of v.
                    neighbors = np.nonzero(adj[v])[0]
                    if len(neighbors) > 0:
                        adj_labels[neighbors] += (1 - 2 * l)
        
        return labels, current_cut

    best_labels = None
    best_cut = -1

    # Multi-start with Spectral Perturbations
    # We use the Fiedler vector as the base, and add small random perturbations
    # to create diverse initial solutions that are still guided by the graph structure.
    
    for trial in range(10):
        if trial == 0:
            # Pure Spectral
            # Threshold at 0 or median
            # Using median ensures balanced partition roughly, but 0 is standard for Fiedler
            # Let's use 0 threshold, but if all signs are same, flip
            labels = (fiedler_vec > 0).astype(np.int8)
            # Check if trivial (all 0 or all 1)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, n, dtype=np.int8)
        else:
            # Perturbed Spectral
            # Add small noise
            noise = np.random.randn(n) * 0.1
            perturbed_vec = fiedler_vec + noise
            labels = (perturbed_vec > 0).astype(np.int8)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, n, dtype=np.int8)
        
        # Local Search
        labels_ls, cut_ls = local_search(labels, max_iters=5000)
        
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
