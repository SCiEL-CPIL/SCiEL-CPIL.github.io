"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Initialization and Iterated Local Search.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph.
        
    Returns:
        np.array of 0/1 labels per vertex.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    total_edges = degrees.sum() // 2
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        # Vectorized cut size calculation
        # crossing[v] = sum_u adj[v,u] * (labels[v] != labels[u])
        # Sum of crossing[v] / 2
        # labels_diff = 1 - 2*abs(labels - 0.5) ... no
        # For binary: labels[v] != labels[u] is 1 if they differ.
        # Let's use the matrix multiplication trick:
        # sum_{v,u} adj[v,u] * (labels[v] - labels[u])^2 / 4 ? No.
        # Standard trick:
        # Cut = (1/4) * sum_{i,j} A_{ij} (1 - 2 x_i)(1 - 2 x_j) ? No.
        # Let's stick to the efficient one:
        # adj_labels = adj @ labels
        # crossing = labels * degrees + adj_labels - 2 * labels * adj_labels
        adj_labels = adj @ labels
        crossing = labels * degrees + adj_labels - 2 * labels * adj_labels
        return int(np.sum(crossing) / 2)
    
    def flip_gain(v, labels, adj_labels):
        """
        Compute the change in cut size if we flip vertex v.
        delta = (1 - 2*labels[v]) * (degrees[v] - 2*adj_labels[v])
        """
        l = labels[v]
        s = adj_labels[v]
        d = degrees[v]
        return (1 - 2 * l) * (d - 2 * s)
    
    def local_search(labels, max_iters=20):
        """Greedy local search: keep flipping vertices that improve the cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        adj_labels = adj @ labels
        
        # We will iterate through vertices multiple times until no improvement
        for _ in range(max_iters):
            improved = False
            for v in range(n):
                delta = flip_gain(v, labels, adj_labels)
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
                    # Update adj_labels for neighbors of v
                    # adj_labels[u] += adj[u,v] * 2 * (1 - 2*old_label) ... 
                    # Actually, if we flip v, adj_labels[u] changes by 2*adj[u,v] * (1 - 2*old_label)?
                    # old_label = 1 - labels[v]
                    # new_label = labels[v]
                    # change in labels[v] is 1 - 2*old_label = 1 - 2*(1-new_label) = 2*new_label - 1
                    # adj_labels[u] = sum_w adj[u,w] labels[w]
                    # delta_adj_labels[u] = adj[u,v] * delta_labels[v]
                    # delta_labels[v] = 1 - 2*old_label = 2*new_label - 1
                    # Since labels[v] is now the new label:
                    # delta_labels = 2*labels[v] - 1
                    # But we need to update adj_labels for all neighbors u of v.
                    # adj_labels[u] += adj[u,v] * (2*labels[v] - 1)
                    # Note: adj is symmetric.
                    # We can do: adj_labels[adj[v] == 1] += 2 * labels[v] - 1
                    adj_labels[adj[v] == 1] += 2 * labels[v] - 1
            if not improved:
                break
        return labels, current_cut

    # 1. Spectral Initialization
    # Build sparse Laplacian
    L = laplacian(csr_matrix(adj))
    
    # Get Fiedler vector (second smallest eigenvalue/eigenvector)
    # For disconnected graphs, eigsh might fail or return zeros for k=0,1.
    # We assume connected or handle by falling back to random.
    try:
        # k=2 gives the two smallest eigenvalues. We want the second one (index 1).
        # sigma='SM' for smallest magnitude.
        # For Laplacian, smallest is 0 (constant vector). Next is Fiedler.
        vals, vecs = eigsh(L, k=2, sigma=0.0, which='LM') 
        # Note: eigsh with sigma=0 can be unstable. Better to use which='SM' if available?
        # scipy.sparse.linalg.eigsh doesn't support sigma for SM directly in all versions?
        # Actually, 'SM' is supported.
        vals, vecs = eigsh(L, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
        
        # Initialize based on sign of Fiedler vector
        labels = (fiedler_vec > 0).astype(int)
        
        # If all labels are same, flip some
        if labels.sum() == 0 or labels.sum() == n:
             labels = np.random.randint(0, 2, n)
            
    except Exception:
        labels = np.random.randint(0, 2, n)
    
    # 2. Iterated Local Search
    best_labels, best_cut = local_search(labels, max_iters=50)
    
    start_time = time.time()
    time_limit = 100.0 # Leave buffer for overhead
    
    while time.time() - start_time < time_limit:
        # Perturb the best solution
        perturbed_labels = best_labels.copy()
        
        # Randomly flip a small number of vertices (e.g., 5-10)
        num_flips = random.randint(3, 10)
        if n > num_flips:
            vertices_to_flip = random.sample(range(n), num_flips)
            for v in vertices_to_flip:
                perturbed_labels[v] = 1 - perturbed_labels[v]
        
        # Local search on perturbed solution
        new_labels, new_cut = local_search(perturbed_labels, max_iters=50)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
