"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # Use matrix multiplication for efficiency
    # For each edge (i,j), it crosses if labels[i] != labels[j]
    # cut = sum over i<j of adjacency[i,j] * (labels[i] != labels[j])
    # Using: cut = 0.5 * sum_{i,j} adjacency[i,j] * (labels[i] - labels[j])^2 / (1) ... 
    # Actually simpler: 
    # Let S0 = set of nodes with label 0, S1 = set with label 1
    # cut = sum_{i in S0, j in S1} adjacency[i,j]
    # We can compute this as: for each node i, count neighbors in the other set
    # Using vectorization:
    diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    return np.sum(adjacency * diff) / 2.0


def _local_search(adjacency, labels, time_budget):
    """Local search with single-flip moves until no improvement or time runs out."""
    n = len(labels)
    labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    # Precompute adjacency as float for faster operations
    adj = adjacency.astype(np.float64)
    
    # Precompute current cut contribution per node
    # For node v, if we flip it, the change in cut is:
    # new_cut - old_cut = (neighbors_in_same_set - neighbors_in_other_set)
    # because edges to same set become crossing, and edges to other set stop crossing
    
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        # Order: random or sequential? Sequential with multiple passes is fine.
        # Let's do a few passes
        for _ in range(3):
            if time.time() >= time_budget:
                break
            for v in range(n):
                # Compute gain from flipping v
                same_count = adj[v, labels == labels[v]].sum()
                other_count = adj[v, labels != labels[v]].sum()
                gain = other_count - same_count  # if gain > 0, flipping improves cut
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
                    best_cut += gain
    return labels, best_cut


def _spectral_init(adjacency, n):
    """Get initial partition from Fiedler vector of graph Laplacian."""
    # Build Laplacian
    degree = np.sum(adjacency, axis=1)
    L = np.diag(degree) - adjacency.astype(np.float64)
    
    # For disconnected graphs, the smallest eigenvalue is 0 with multiplicity > 1
    # We want the second smallest eigenvector (Fiedler vector)
    # Use scipy if available, otherwise numpy
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import eigsh
        L_sparse = csc_matrix(L)
        # Find the two smallest eigenvalues
        vals, vecs = eigsh(L_sparse, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    except ImportError:
        # Fallback to numpy
        vals, vecs = np.linalg.eigh(L)
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    
    # Partition based on median of Fiedler vector
    median_val = np.median(fiedler)
    labels = (fiedler > median_val).astype(np.int32)
    return labels


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Convert to numpy array once
    adj = np.array(adjacency, dtype=np.float64)
    
    # Multi-start: combine spectral init with random inits
    best_labels = None
    best_cut = -1
    
    # Start 1: Spectral initialization
    labels_spec = _spectral_init(adj, n)
    labels_ref, cut_val = _local_search(adj, labels_spec, deadline)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = labels_ref.copy()
    
    # If time remains, do additional random restarts
    rng = np.random.default_rng(42)
    while time.time() < deadline - 5.0:  # Leave 5s buffer
        # Random initialization
        labels_rand = rng.integers(0, 2, size=n).astype(np.int32)
        labels_ref, cut_val = _local_search(adj, labels_rand, deadline)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels_ref.copy()
    
    # Final polish: try a few more local search passes on the best solution
    if time.time() < deadline - 1.0:
        labels_final, cut_final = _local_search(adj, best_labels, deadline)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final
    
    return best_labels
# EVOLVE-BLOCK-END
