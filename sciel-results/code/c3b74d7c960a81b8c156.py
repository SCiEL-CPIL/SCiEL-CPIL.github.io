# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1  # ±1
    E_total = 0.5 * adj.sum()
    s_As = s @ (adj @ s)
    cut = E_total - 0.25 * s_As
    return int(round(cut))


def _greedy_refine(adj, labels, max_passes=100, deadline=None):
    """
    Flip-based greedy local search with incremental updates.
    Uses a queue-based approach for efficiency.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2 * labels.astype(np.float64) - 1
    
    # Compute (A s)
    As = adj @ s
    
    # gain[v] = 0.5 * s[v] * As[v]
    gain = 0.5 * s * As
    
    # Find initially positive gains
    positive = np.nonzero(gain > 1e-9)[0]
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        if len(positive) == 0:
            break
        
        # Process vertices with positive gain in random order
        np.random.shuffle(positive)
        
        # Re-find positive gains each pass since flips change things
        # Actually, let's do a single pass through all vertices, flipping if beneficial
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            if gain[v] > 1e-9:
                # Flip v
                s[v] = -s[v]
                labels[v] = 1 - labels[v]
                improved = True
                # Update As for all neighbors of v
                delta = 2.0 * s[v]
                As += delta * adj[v]
                # Update gain for v and its neighbors
                gain[v] = 0.5 * s[v] * As[v]
                neighbors = np.nonzero(adj[v])[0]
                if len(neighbors) > 0:
                    gain[neighbors] = 0.5 * s[neighbors] * As[neighbors]
        
        if not improved:
            break
    
    return labels


def _simulated_annealing(adj, labels, deadline, rng, T0=10.0, cooling=0.99, min_T=0.01, max_iters=50000):
    """
    Simulated annealing to escape local optima.
    Returns improved labels.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2 * labels.astype(np.float64) - 1
    As = adj @ s
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T = T0
    iters = 0
    while T > min_T and iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1
        
        # Randomly pick a vertex to flip
        v = rng.integers(0, n)
        
        # Compute gain of flipping v
        gain_v = 0.5 * s[v] * As[v]
        
        # Accept with probability 1 if gain > 0, else exp(gain/T)
        if gain_v > 0 or (T > 0 and rng.random() < np.exp(gain_v / T)):
            # Flip v
            s[v] = -s[v]
            labels[v] = 1 - labels[v]
            current_cut += gain_v
            
            # Update As
            delta = 2.0 * s[v]
            As += delta * adj[v]
            
            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= cooling
    
    return best_labels


def _spectral_init(adj, n):
    """
    Use the Fiedler vector (2nd smallest eigenvector of Laplacian) to initialize.
    """
    degree = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = sp.diags(degree)
    A_sparse = sp.csr_matrix(adj)
    L = D - A_sparse
    
    try:
        k = min(4, n - 1)
        if k < 2:
            return np.arange(n) % 2
        evals, evecs = spla.eigsh(L, k=k, which='SM')
        fiedler = evecs[:, 1]
        labels = (fiedler > 0).astype(int)
        return labels
    except Exception:
        return np.arange(n) % 2


def _sdp_relaxation(adj, n, deadline, rng, n_iters=2000):
    """
    SDP relaxation via projected gradient ascent with Goemans-Williamson rounding.
    Maximizes trace(X * L/2) where L is Laplacian, X = s s^T for s in S^{n-1}.
    """
    degree = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = sp.diags(degree)
    A_sparse = sp.csr_matrix(adj)
    L = D - A_sparse
    L_dense = L.toarray()
    
    # Initialize X as random orthonormal vectors
    X = np.zeros((n, n))
    for i in range(n):
        X[i, i] = 1.0
    
    # Use random initial vectors
    U, _, Vt = np.linalg.svd(np.random.randn(n, n))
    X = U @ U.T
    # Project to correlation matrix: X = U U^T is already PSD with unit diagonal
    # But we need X[i,i] = 1 exactly
    X = (X + X.T) / 2
    np.fill_diagonal(X, 1.0)
    
    best_X = X.copy()
    best_obj = np.trace(X @ L_dense) / 2.0
    
    # Projected gradient ascent
    step_size = 2.0 / np.linalg.norm(L_dense, ord=2) if np.linalg.norm(L_dense, ord=2) > 0 else 1.0
    step_size = min(step_size, 1.0)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Gradient of trace(X L) / 2 is L
        grad = L_dense
        
        # Update
        X_new = X + step_size * grad
        
        # Project to correlation matrix: X PSD, X[i,i]=1
        # Use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(X_new)
        eigvals = np.maximum(eigvals, 0)
        X_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        X_new = (X_new + X_new.T) / 2
        np.fill_diagonal(X_new, 1.0)
        
        # Check objective
        obj = np.trace(X_new @ L_dense) / 2.0
        if obj > best_obj:
            best_obj = obj
            best_X = X_new.copy()
        
        X = X_new
    
    # Goemans-Williamson rounding
    # Random hyperplane: sample u from standard normal, label by sign of X u
    u = rng.standard_normal(n)
    labels = (X @ u > 0).astype(int)
    
    return labels


def _pairwise_swap_search(adj, labels, deadline, rng, max_swaps=20000):
    """
    Pairwise swap local search: try swapping labels of pairs of vertices.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2 * labels.astype(np.float64) - 1
    As = adj @ s
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Pick two random vertices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Compute gain of swapping i and j
        # Flipping both i and j
        # Gain = 0.5 * s[i] * As[i] + 0.5 * s[j] * As[j] - cross term
        # When we flip i: As changes by 2*s[i]*adj[i]
        # When we flip j: As changes by 2*s[j]*adj[j]
        # Combined effect on cut:
        # The change in s^T A s when flipping i and j:
        # Let's compute directly
        s_i = s[i]
        s_j = s[j]
        
        # Change in s^T A s:
        # Original contribution of i and j to s^T A s:
        # s_i * (As[i]) + s_j * (As[j])
        # After flip: (-s_i) * (As[i] + 2*s_i*adj[i] + 2*s_j*adj[j]) + (-s_j) * (As[j] + 2*s_j*adj[j] + 2*s_i*adj[i])
        # This is complex. Let's just compute the cut difference directly.
        
        # Simpler: compute cut before and after
        s_new = s.copy()
        s_new[i] = -s_new[i]
        s_new[j] = -s_new[j]
        
        As_new = adj @ s_new
        cut_new = 0.5 * adj.sum() - 0.25 * (s_new @ As_new)
        
        if cut_new > best_cut:
            # Accept
            s = s_new
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            As = As_new
            current_cut = cut_new
            best_cut = cut_new
            best_labels = labels.copy()
    
    return best_labels


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + SDP Relaxation + Greedy Local Search + 
    Simulated Annealing + Pairwise Swap Search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = 0.5 * (adj + adj.T)
    
    rng = np.random.default_rng(42)
    
    # Collect multiple initializations
    inits = []
    
    # 1. Spectral initialization
    labels_spec = _spectral_init(adj, n)
    inits.append(labels_spec)
    
    # 2. SDP relaxation initialization
    sdp_deadline = deadline - 60  # leave time for other methods
    labels_sdp = _sdp_relaxation(adj, n, sdp_deadline, rng, n_iters=1500)
    inits.append(labels_sdp)
    
    # 3. Random initializations
    for r in range(3):
        if time.time() > deadline - 30:
            break
        labels_rand = rng.integers(0, 2, size=n)
        inits.append(labels_rand)
    
    # Refine each initialization with greedy local search
    best_labels = None
    best_cut = -1
    
    for labels_init in inits:
        if time.time() > deadline - 20:
            break
        labels_ref = _greedy_refine(adj, labels_init, max_passes=80, deadline=deadline - 15)
        cut_ref = _cut_size(adj, labels_ref)
        if cut_ref > best_cut:
            best_cut = cut_ref
            best_labels = labels_ref
    
    # Simulated annealing on the best solution
    if time.time() < deadline - 10:
        sa_deadline = deadline - 5
        sa_labels = _simulated_annealing(adj, best_labels, sa_deadline, rng, T0=10.0, cooling=0.99, min_T=0.01, max_iters=30000)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Pairwise swap search
    if time.time() < deadline - 5:
        swap_deadline = deadline - 2
        swap_labels = _pairwise_swap_search(adj, best_labels, swap_deadline, rng, max_swaps=15000)
        swap_cut = _cut_size(adj, swap_labels)
        if swap_cut > best_cut:
            best_cut = swap_cut
            best_labels = swap_labels
    
    # Final polish: one more greedy pass on the best
    if time.time() < deadline - 1:
        polished = _greedy_refine(adj, best_labels, max_passes=100, deadline=deadline - 0.5)
        polished_cut = _cut_size(adj, polished)
        if polished_cut > best_cut:
            best_labels = polished
            best_cut = polished_cut
    
    return best_labels
# EVOLVE-BLOCK-END
