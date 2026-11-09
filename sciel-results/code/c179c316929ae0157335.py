"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def _cut_size(labels, adj):
    """Compute the number of edges crossing the partition."""
    # edges crossing: one endpoint in 0, other in 1
    # For each edge (i,j) where i<j, count if labels[i] != labels[j]
    # Efficient: sum over all i,j where labels differ and adj[i,j]=1, divide by 2
    # Or: for each vertex, count neighbors in opposite partition
    # cut = sum over v of (number of neighbors of v in opposite partition) / 2
    # But since we're using 0/1 labels:
    # For vertex v with label l, its contribution is sum of adj[v, labels != l]
    # Total cut = 0.5 * sum over all v of (neighbors in opposite partition)
    
    # Vectorized approach:
    # cut = 0.5 * (sum over i,j of adj[i,j] * (labels[i] != labels[j]))
    # labels[i] != labels[j] is 1 when they differ
    
    # Using matrix operations:
    # Let's compute: for each vertex, count neighbors with different label
    # Then sum and divide by 2
    
    # adj is the boolean/float matrix
    # labels is 0/1 array
    # neighbors_in_opposite[v] = sum of adj[v, :] where labels != labels[v]
    
    # This can be done as:
    # For label 0 vertices: count neighbors with label 1
    # For label 1 vertices: count neighbors with label 0
    
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)
    
    # For vertices with label 0, count neighbors with label 1
    # For vertices with label 1, count neighbors with label 0
    
    # Using matrix multiplication or element-wise:
    # cut = 0.5 * (adj @ mask_1.astype(float)).sum() + 0.5 * (adj @ mask_0.astype(float)).sum()
    # Wait, that's not quite right. Let me think again.
    
    # Actually: cut = sum over all edges (i,j) where labels[i] != labels[j]
    # = 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
    
    # (labels[i] != labels[j]) = 1 - (labels[i] == labels[j])
    # = 1 - labels[i]*labels[j] - (1-labels[i])*(1-labels[j])
    # = 1 - labels[i]*labels[j] - 1 + labels[i] + labels[j] - labels[i]*labels[j]
    # = labels[i] + labels[j] - 2*labels[i]*labels[j]
    
    # So cut = 0.5 * sum_{i,j} adj[i,j] * (labels[i] + labels[j] - 2*labels[i]*labels[j])
    # = 0.5 * (sum_{i,j} adj[i,j]*labels[i] + sum_{i,j} adj[i,j]*labels[j] - 2*sum_{i,j} adj[i,j]*labels[i]*labels[j])
    # = 0.5 * (2 * sum_i labels[i] * sum_j adj[i,j] - 2 * sum_{i,j} adj[i,j]*labels[i]*labels[j])
    # = sum_i labels[i] * degree[i] - sum_{i,j} adj[i,j]*labels[i]*labels[j]
    
    # Let deg = adj.sum(axis=1)
    # Let L = labels as float
    # cut = L @ deg - L @ adj @ L
    
    L = labels.astype(np.float64)
    deg = adj.sum(axis=1)
    cut = L @ deg - L @ adj @ L
    return cut


def _local_search(labels, adj, deadline=None):
    """Greedy local search: flip vertices that improve the cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for v in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping vertex v
            # Compute the change in cut if we flip v
            # Current contribution of v to cut:
            #   neighbors of v with opposite label
            # After flip:
            #   neighbors of v with same label (which become opposite)
            
            # Let's compute directly:
            old_label = labels[v]
            new_label = 1 - old_label
            
            # Neighbors of v
            neighbors = adj[v]
            
            # Count neighbors with label != old_label (these are currently crossing)
            crossing_now = (neighbors * (labels != old_label)).sum()
            # Count neighbors with label == old_label (these will become crossing after flip)
            becoming_crossing = (neighbors * (labels == old_label)).sum()
            
            # Change in cut: -crossing_now + becoming_crossing
            # But we also need to account for the fact that v's own label changes
            # Actually, the cut size is symmetric, so flipping v changes the cut by:
            # (number of neighbors with same label as v) - (number of neighbors with different label as v)
            
            if becoming_crossing > crossing_now:
                labels[v] = new_label
                current_cut += (becoming_crossing - crossing_now)
                improved = True
    
    return labels, current_cut


def _spectral_init(adj, n):
    """Initialize using the Fiedler vector (second smallest eigenvector of Laplacian)."""
    # Compute Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Find the second smallest eigenvector
    # Use scipy's eigsh for sparse eigenvalue problem
    try:
        # k=2 to get the two smallest eigenvalues
        # sigma=0 to find eigenvalues near 0
        vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    except Exception:
        # Fallback to random
        fiedler = np.random.randn(n)
    
    # Use the sign of the Fiedler vector to create initial partition
    # Vertices with positive Fiedler value go to one side, negative to the other
    labels = (fiedler >= 0).astype(np.int32)
    
    # Ensure both sides are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        # Fallback
        labels = np.arange(n) % 2
    
    return labels


def _random_init(n):
    """Random initialization."""
    return np.random.randint(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple initializations
    inits = []
    
    # 1. Spectral initialization
    try:
        spectral_labels = _spectral_init(adj, n)
        inits.append(spectral_labels)
    except Exception:
        pass
    
    # 2. Random initializations
    for _ in range(3):
        inits.append(_random_init(n))
    
    # 3. Alternating initialization (from original)
    inits.append(np.arange(n) % 2)
    
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        # Run local search
        labels, cut = _local_search(init_labels, adj, deadline)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we didn't get any valid result, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
