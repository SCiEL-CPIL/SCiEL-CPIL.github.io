"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array if not already
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Build Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Solve L * x = 0 for the null space (should be constant vector for connected graph)
    # We want the eigenvectors corresponding to the smallest non-zero eigenvalues
    # Use sparse eigenvalue solver for efficiency
    
    # Convert to sparse matrix
    L_sparse = sp.csr_matrix(L)
    
    # Find k smallest eigenvalues (excluding the trivial 0 eigenvalue)
    # We'll find the first few non-trivial eigenvectors
    k = min(10, n - 1)
    
    try:
        # Find smallest eigenvalues
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, which='SM', return_eigenvectors=True)
        
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Use the eigenvectors as the embedding coordinates
        # Skip the first eigenvector if it's the trivial constant one
        # The trivial eigenvector corresponds to eigenvalue 0
        non_trivial_mask = eigvals > 1e-8
        if np.sum(non_trivial_mask) > 0:
            embedding = eigvecs[:, non_trivial_mask]
        else:
            embedding = eigvecs
        
    except Exception:
        # Fallback to simple alternating pattern if eigensolver fails
        labels = np.arange(n) % 2
        # Do a few rounds of local improvement
        for _ in range(5):
            improved = False
            for v in range(n):
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
            if not improved:
                break
        return labels.astype(int)
    
    # Now perform randomized hyperplane projection
    # Generate several random hyperplanes and use them to partition
    best_labels = None
    best_cut = -1
    
    n_hyperplanes = 20
    
    for trial in range(n_hyperplanes):
        # Generate a random unit vector in the embedding space
        dim = embedding.shape[1]
        random_vec = np.random.randn(dim)
        random_vec /= np.linalg.norm(random_vec)
        
        # Project embedding onto this hyperplane
        projections = embedding @ random_vec
        
        # Partition: positive projections -> 1, negative -> 0
        labels = (projections >= 0).astype(int)
        
        # Ensure both partitions are non-empty (if possible)
        if labels.sum() == 0 or labels.sum() == n:
            # Flip a few random vertices
            flip_idx = np.random.choice(n, size=max(1, n // 10), replace=False)
            labels[flip_idx] = 1 - labels[flip_idx]
        
        # Local improvement
        labels = local_improve(adj, labels, n, max_rounds=10)
        
        # Compute cut size
        cut_size = compute_cut_size(adj, labels)
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_labels = local_improve(adj, best_labels, n, max_rounds=10)
    
    return best_labels.astype(int)


def compute_cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Edge crosses if labels differ
    cross = (labels[:, None] != labels[None, :])
    cut_size = adj.sum() / 2 * 0  # Start at 0
    cut_size = (adj * cross).sum() / 2
    return cut_size


def local_improve(adj, labels, n, max_rounds=10):
    """Perform local search to improve the partition."""
    labels = labels.copy()
    
    for _ in range(max_rounds):
        improved = False
        # Randomize the order of vertices for diversity
        order = np.random.permutation(n)
        for v in order:
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
