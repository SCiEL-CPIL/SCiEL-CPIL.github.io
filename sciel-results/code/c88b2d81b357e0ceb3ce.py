"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)


def _spectral_init(adj, n):
    """
    Initialize using the second eigenvector of the graph Laplacian.
    Returns a 0/1 label array.
    """
    # Build Laplacian L = D - A
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        # Use numpy's eigvalsh/eigh for symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvector (index 1) is the Fiedler vector
        fiedler = eigenvectors[:, 1]
        # Threshold at median to create balanced partition
        median_val = np.median(fiedler)
        labels = (fiedler >= median_val).astype(np.int8)
        return labels
    except Exception:
        # Fallback to random if spectral fails
        return np.random.randint(0, 2, size=n)


def _goemans_maxwell(adj, n):
    """
    Goemans-Williamson style randomized rounding using random hyperplane.
    Returns a 0/1 label array.
    """
    # Build Laplacian
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    # Get eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except Exception:
        return np.random.randint(0, 2, size=n)

    # Random direction in the space spanned by eigenvectors
    # We can use a random vector in R^n and project
    random_vec = np.random.randn(n)
    # Project onto the null space of L (which is spanned by 1-vector)
    # Actually, for the cut problem, we want to use the eigenvectors
    # Let's use a random linear combination of the top eigenvectors
    # Actually, simpler: just use a random vector and threshold
    # But the Goemans-Williamson approach uses the eigenvectors
    
    # Better approach: use random hyperplane in the eigenspace
    # Generate random coefficients for each eigenvector
    coeffs = np.random.randn(len(eigenvalues))
    # Weight by inverse of eigenvalue (or just use all)
    # Actually, the standard approach: v = sum_i c_i * u_i where u_i are eigenvectors
    # Then threshold v at 0
    
    # But we need to be careful with the null space
    # The first eigenvector is the all-ones vector (eigenvalue 0)
    # Let's skip it and use the rest
    v = np.zeros(n)
    for i in range(1, n):
        v += coeffs[i] * eigenvectors[:, i]
    
    # Threshold at 0
    labels = (v >= 0).astype(np.int8)
    return labels


def _greedy_improve(adj, labels, deadline):
    """Iteratively flip vertices that improve the cut until local optimum or deadline."""
    n = len(labels)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    deg = adj_int.sum(axis=1)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for v in range(n):
            if time.time() > deadline:
                return labels
            same = adj_int[v][labels == labels[v]].sum()
            other = deg[v] - same
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _simulated_annealing(adj, labels, deadline, iterations=2000):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    deg = adj_int.sum(axis=1)
    
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Temperature schedule
    t_start = time.time()
    total_time = deadline - t_start
    if total_time <= 0:
        return best_labels
    
    # Start with high temperature, cool down
    T = 5.0
    cooling_rate = 0.999
    
    for i in range(iterations):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(0, n)
        old_label = labels[v]
        new_label = 1 - old_label
        
        # Compute delta cut
        same = adj_int[v][labels == old_label].sum()
        other = deg[v] - same
        # If we flip, the new same becomes other, new other becomes same
        # Delta = new_crossing - old_crossing for vertex v
        # old contribution: other (edges to opposite side)
        # new contribution: same (edges to what was same side, now opposite)
        delta = same - other
        
        # Accept with probability
        if delta > 0:
            # Always accept improvement
            labels[v] = new_label
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    labels[v] = new_label
                    current_cut += delta
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
        
        # Cool down
        T *= cooling_rate
    
    return best_labels


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    # Use a safety margin: stop searching 10 seconds before the 120s limit
    deadline = start_time + 110.0
    
    # Convert to numpy array once
    adj = np.asarray(adjacency, dtype=np.int8)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + greedy
    if time.time() < deadline:
        labels_spec = _spectral_init(adj, n)
        labels_spec = _greedy_improve(adj, labels_spec, deadline)
        cut = _cut_size(adj, labels_spec)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_spec.copy()
    
    # Strategy 2: Goemans-Williamson style + greedy
    if time.time() < deadline:
        labels_gw = _goemans_maxwell(adj, n)
        labels_gw = _greedy_improve(adj, labels_gw, deadline)
        cut = _cut_size(adj, labels_gw)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_gw.copy()
    
    # Strategy 3: Random restarts with greedy
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
        labels_rand = np.random.randint(0, 2, size=n)
        labels_rand = _greedy_improve(adj, labels_rand, deadline)
        cut = _cut_size(adj, labels_rand)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_rand.copy()
    
    # Strategy 4: Simulated annealing from best found so far
    if time.time() < deadline - 10 and best_labels is not None:
        labels_sa = _simulated_annealing(adj, best_labels, deadline, iterations=3000)
        cut = _cut_size(adj, labels_sa)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_sa.copy()
    
    # Final greedy polish
    if best_labels is not None and time.time() < deadline - 2:
        best_labels = _greedy_improve(adj, best_labels, deadline)
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
        best_labels = _greedy_improve(adj, best_labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
