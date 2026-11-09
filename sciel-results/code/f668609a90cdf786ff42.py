"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # cut = sum_{v<u} adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * (sum_v adj[v] @ (1 - labels)) where labels are 0/1
    # More precisely: for each v, count neighbors u with labels[u] != labels[v]
    # Using matrix ops:
    # Let x = labels (0/1). Then cut = 0.5 * x^T (D - A) (1-x) ... 
    # Simpler: cut = sum_v (adj[v] @ (1 - labels)) / 2? No.
    # Actually: cut = 0.5 * sum_{v,u} adj[v,u] * (labels[v] - labels[u])^2
    # Since (labels[v]-labels[u])^2 = 1 if different, 0 if same.
    # = 0.5 * (sum_v labels[v]^2 * deg[v] + sum_u labels[u]^2 * deg[u] - 2 * sum_{v,u} adj[v,u]*labels[v]*labels[u])
    # = 0.5 * (sum_v labels[v]*deg[v] + sum_u labels[u]*deg[u] - 2*labels^T @ adj @ labels)
    # Since labels are 0/1, labels[v]^2 = labels[v]
    # = 0.5 * (2 * labels @ deg - 2 * labels @ adj @ labels)
    # = labels @ deg - labels @ adj @ labels
    
    deg = adj.sum(axis=1)
    return float(labels @ deg - labels @ adj @ labels)


def _spectral_init(adj):
    """Initialize labels using the Fiedler vector (second eigenvector of Laplacian)."""
    n = len(adj)
    deg = adj.sum(axis=1)
    # Laplacian L = D - A
    # We want the second smallest eigenvalue's eigenvector.
    # Use scipy if available, else use power iteration on shifted matrix.
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        L = diags(deg) - adj
        # eigsh finds eigenvalues near sigma
        # We want the smallest eigenvalue (0) and the next one.
        # Use sigma=0 to find smallest, but we need the second.
        # Alternatively, compute eigenvalues of L directly for small n.
        # For n=100, dense eigendecomposition is fine.
        L_dense = np.diag(deg) - adj
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Second eigenvector (index 1)
        fiedler = eigvecs[:, 1]
        # Threshold at median
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(float)
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except Exception:
        # Fallback: alternating
        return np.arange(n) % 2


def _greedy_flip_improve(adj, labels, max_iters=100):
    """Improve labels by flipping vertices that increase cut size."""
    n = len(adj)
    deg = adj.sum(axis=1)
    current_cut = _cut_size(adj, labels)
    
    for _ in range(max_iters):
        improved = False
        for v in range(n):
            # If we flip v, what is the change in cut?
            # Current contribution of v to cut: adj[v] @ (1 - labels)
            # After flip, labels[v] changes, so contribution becomes adj[v] @ (1 - new_labels)
            # But also, for all u, the term involving (labels[u]-labels[v]) changes.
            # Easier: just compute new cut after flip.
            old_label = labels[v]
            labels[v] = 1 - old_label
            new_cut = _cut_size(adj, labels)
            if new_cut > current_cut:
                current_cut = new_cut
                improved = True
            else:
                labels[v] = old_label
        if not improved:
            break
    
    return labels, current_cut


def _two_swap_improve(adj, labels, max_iters=50):
    """Improve labels by swapping pairs of vertices."""
    n = len(adj)
    current_cut = _cut_size(adj, labels)
    
    for _ in range(max_iters):
        improved = False
        # Try random pairs
        for _ in range(200):
            v1, v2 = random.sample(range(n), 2)
            if labels[v1] == labels[v2]:
                continue
            # Swap labels
            labels[v1], labels[v2] = labels[v2], labels[v1]
            new_cut = _cut_size(adj, labels)
            if new_cut > current_cut:
                current_cut = new_cut
                improved = True
            else:
                # Swap back
                labels[v1], labels[v2] = labels[v2], labels[v1]
        if not improved:
            break
    
    return labels, current_cut


def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    adj = np.array(adjacency, dtype=float)
    n = len(adj)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Spectral initialization
            labels = _spectral_init(adj)
        elif restart % 3 == 1:
            # Random initialization
            labels = np.random.randint(0, 2, size=n).astype(float)
        else:
            # Alternating with random perturbation
            labels = np.arange(n) % 2
            # Randomly flip some vertices
            flip_mask = np.random.random(n) < 0.2
            labels[flip_mask] = 1 - labels[flip_mask]
        
        # Greedy flip improvement
        labels, cut = _greedy_flip_improve(adj, labels, max_iters=50)
        
        # Two-swap improvement
        labels, cut = _two_swap_improve(adj, labels, max_iters=30)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
