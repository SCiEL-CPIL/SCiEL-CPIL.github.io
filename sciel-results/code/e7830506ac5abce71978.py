# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    n = len(labels)
    # For each edge (i,j) where adj[i,j]=1, count if labels differ
    # Vectorized: create matrix of label differences
    diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    return 0.5 * np.sum(adj * diff)  # 0.5 because each edge counted twice


def _spectral_init(adj, n):
    """Initialize partition using the eigenvector corresponding to the largest eigenvalue."""
    # For max cut, we want to maximize s^T (J - A) s / 4 where s is ±1
    # Equivalent to minimizing s^T A s
    # The leading eigenvector of A gives a good starting point
    # Actually, for max cut we want to maximize the cut, which relates to
    # the eigenvector of the Laplacian L = D - A
    
    # Simpler: use the second eigenvector of the adjacency matrix
    # or use the leading eigenvector of (J - A) which is the max-cut relaxation
    
    # Let's use the approach: maximize x^T A x where x is ±1
    # The relaxation gives the eigenvector of A with largest eigenvalue
    # But we want to MINIMIZE x^T A x (since cut = E - same_side_edges)
    
    # Actually: cut = (E - sum_{ij} A_ij * s_i * s_j) / 2 where s = ±1
    # So we want to minimize s^T A s
    # The minimum is given by the eigenvector with the most negative eigenvalue
    
    # For a general graph, let's just use the leading eigenvector of A
    # and threshold it
    try:
        # Use sparse eigensolver for efficiency
        A = sp.csr_matrix(adj)
        # Get the eigenvector corresponding to the largest magnitude eigenvalue
        # For max cut, we want to separate nodes, so use the second eigenvector
        # of the Laplacian, or just use random + local search
        
        # Simple approach: use the eigenvector of A
        k = 2
        vals, vecs = spla.eigsh(A, k=k, which='LM')
        # Use the eigenvector with the largest eigenvalue
        idx = np.argmax(np.abs(vals))
        v = vecs[:, idx]
        
        # Threshold at median to get balanced partition
        median_val = np.median(v)
        labels = (v > median_val).astype(np.int64)
        
        # Ensure both classes are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2


def _greedy_refine(adj, labels, max_passes=50):
    """Greedy local improvement: flip a vertex if it increases the cut."""
    n = len(labels)
    adj = np.asarray(adj)
    
    # Precompute degrees and current cut contribution
    best_cut = _cut_size(adj, labels)
    
    for _pass in range(max_passes):
        improved = False
        # For each vertex, compute the gain from flipping
        # If we flip vertex v, the change in cut is:
        # new_cut - old_cut = (neighbors_in_other - neighbors_in_same) 
        # where neighbors_in_other = number of neighbors with different label
        # and neighbors_in_same = number of neighbors with same label
        
        # Vectorized computation:
        # For vertex v, gain = sum_{u != v} adj[v,u] * (1 if labels[v] != labels[u] else -1)
        # Actually, if we flip v:
        # - Edges from v to same-side neighbors become crossing (+1 each)
        # - Edges from v to other-side neighbors become same-side (-1 each)
        # So gain = (same_side_neighbors) - (other_side_neighbors)
        
        # Compute for all vertices at once
        # same_side[v] = sum of adj[v, u] where labels[u] == labels[v]
        # other_side[v] = sum of adj[v, u] where labels[u] != labels[v]
        
        # Create a matrix where M[v,u] = 1 if labels[v] == labels[u], else 0
        same_mask = (labels[:, None] == labels[None, :]).astype(np.float64)
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(same_mask, 0)
        
        same_side = adj @ same_mask  # For each v, sum of adj[v,u] where same label
        other_side = adj.sum(axis=1) - same_side  # Total degree minus same-side
        
        # Gain from flipping vertex v = same_side[v] - other_side[v]
        # We want to flip if gain > 0, i.e., same_side[v] > other_side[v]
        gains = same_side - other_side
        
        # Find vertices with positive gain
        flip_candidates = np.where(gains > 0)[0]
        
        if len(flip_candidates) == 0:
            break
        
        # Flip all candidates simultaneously (or one at a time for correctness)
        # For a more aggressive approach, flip all at once
        # But this might not be optimal; let's flip one at a time in order of gain
        flip_order = flip_candidates[np.argsort(-gains[flip_candidates])]
        
        for v in flip_order:
            # Recompute gain for v considering other flips
            # For simplicity, just flip if original gain was positive
            # This is a heuristic; for better results, recompute
            labels[v] = 1 - labels[v]
            improved = True
        
        if not improved:
            break
    
    return labels, _cut_size(adj, labels)


def partition(adjacency):
    """Find a maximum cut partition of the graph."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = len(adj)
    
    if n == 0:
        return np.array([], dtype=np.int64)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + greedy refinement
    spectral_labels = _spectral_init(adj, n)
    refined_labels, cut_val = _greedy_refine(adj, spectral_labels.copy())
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels
    
    # Strategy 2: Multiple random restarts with greedy refinement
    np.random.seed(42)
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy refinement
        refined_labels, cut_val = _greedy_refine(adj, labels.copy())
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels
    
    # Strategy 3: More thorough local search with multiple passes
    # Try to improve further with additional passes
    final_labels = best_labels.copy()
    for _ in range(5):
        if time.time() > deadline:
            break
        refined_labels, cut_val = _greedy_refine(adj, final_labels.copy(), max_passes=100)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
            final_labels = refined_labels.copy()
        else:
            break
    
    return best_labels
# EVOLVE-BLOCK-END
