import numpy as np
import time
import math

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Vectorized cut size calculation
    # adj is bool matrix. labels is 0/1 array.
    # Crossing edges are where adj[i,j] is True and labels[i] != labels[j].
    # Sum of adj[labels[:, None] != labels[None, :]] // 2
    return int(adj[labels[:, None] != labels[None, :]].sum() // 2)

def _spectral_init(adj):
    """
    Initialize labels using the Fiedler vector (eigenvector of the Laplacian
    corresponding to the second smallest eigenvalue).
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    
    # Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj.astype(float)
    
    try:
        # Find eigenvalues and eigenvectors
        # For n=100, this is fast.
        vals, vecs = np.linalg.eigh(L)
        
        # The smallest eigenvalue is 0 (for connected graph).
        # The second smallest corresponds to the Fiedler vector.
        # eigh returns eigenvalues in ascending order.
        # So vecs[:, 1] is the Fiedler vector.
        fiedler = vecs[:, 1]
        
        # Threshold at 0 (or mean) to get 0/1 labels
        # Usually thresholding at 0 works well.
        labels = (fiedler > 0).astype(int)
        
        # Ensure we have a mix of 0s and 1s if possible, 
        # otherwise fall back to random
        if labels.sum() == 0 or labels.sum() == n:
             labels = np.random.randint(0, 2, n)
             
        return labels
    except Exception:
        return np.random.randint(0, 2, n)

def _greedy_optimize(adj, labels, max_passes=10):
    """
    Greedy local search: flip vertices if it increases the cut size.
    Repeat until no single flip improves the cut.
    """
    n = len(labels)
    adj_int = adj.astype(int)
    current_cut = _cut_size(adj, labels)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            # Calculate gain of flipping v
            # k_same = neighbors with same label
            # k_diff = neighbors with different label
            # Gain = k_same - k_diff
            
            # Vectorized calculation for vertex v
            # neighbors = adj_int[v]
            # same_mask = (labels == labels[v])
            # k_same = np.dot(neighbors, same_mask)
            # k_diff = np.dot(neighbors, ~same_mask)
            
            # Faster: 
            # Let s = labels[v] (0 or 1)
            # neighbors with label s: adj_int[v][labels == s]
            # neighbors with label 1-s: adj_int[v][labels != s]
            
            s = labels[v]
            k_same = np.dot(adj_int[v], labels == s)
            k_diff = np.dot(adj_int[v], labels != s)
            
            delta = k_same - k_diff
            
            if delta > 0:
                # Flip
                labels[v] = 1 - s
                current_cut += delta
                improved = True
        
        if not improved:
            break
            
    return labels, current_cut

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=bool)
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    init_labels = _spectral_init(adj)
    
    # 2. Optimize the spectral init
    opt_labels, opt_cut = _greedy_optimize(adj, init_labels)
    
    if opt_cut > best_cut:
        best_cut = opt_cut
        best_labels = opt_labels.copy()
    
    # 3. Multi-start Local Search with Perturbation
    # Use random restarts or perturbations of the best solution
    # Since n=100, greedy is very fast. We can do many iterations.
    
    while time.time() - start_time < deadline:
        # Strategy: Take current best, perturb it, and re-optimize
        if best_labels is None:
            current = np.random.randint(0, 2, n)
        else:
            current = best_labels.copy()
            
        # Perturb: flip k random vertices
        # k can vary. Let's try small perturbations.
        k = np.random.randint(1, 5) # Flip 1 to 4 vertices
        indices = np.random.choice(n, k, replace=False)
        current[indices] = 1 - current[indices]
        
        # Optimize
        new_labels, new_cut = _greedy_optimize(adj, current)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            
        # Occasionally try a completely random start if time permits
        if np.random.random() < 0.05:
            random_labels = np.random.randint(0, 2, n)
            rand_opt_labels, rand_cut = _greedy_optimize(adj, random_labels)
            if rand_cut > best_cut:
                best_cut = rand_cut
                best_labels = rand_opt_labels.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
        
    return best_labels
