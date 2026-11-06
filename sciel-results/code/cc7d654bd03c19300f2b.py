# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1
    # cut = 0.5 * (sum(A) - s^T A s)
    return 0.5 * (np.sum(adj) - s @ adj @ s)


def _spectral_init(adj, n):
    """Generate initial labels using spectral method (Fiedler vector)."""
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    try:
        evals, evecs = np.linalg.eigh(L)
        fiedler = evecs[:, 1]
    except:
        return np.random.randint(0, 2, n).astype(np.float64)
    
    labels = (fiedler > 0).astype(np.float64)
    
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n).astype(np.float64)
    
    return labels


def _sdr_init(adj, n, deadline, n_iter=2000):
    """
    SDR (Semidefinite Relaxation) initialization using projected gradient ascent.
    Solves max tr(A*W) s.t. W_ii = 1, W >= 0
    Then uses Goemans-Williamson randomized rounding.
    """
    # Initialize W as identity (W_ii = 1)
    W = np.eye(n, dtype=np.float64)
    
    # Gradient of tr(A*W) wrt W is A (since A is symmetric)
    # But we need to project onto {W: W_ii=1, W >= 0}
    # The projection onto diagonal constraint W_ii=1 is just fixing the diagonal
    # The projection onto PSD is via eigendecomposition
    
    # Use projected gradient ascent
    # At each step: W_new = W + step * A
    # Then project: set diagonal to 1, then project to PSD
    
    # For efficiency, we'll use a simple approach:
    # W = V V^T where V is n x n
    # Constraint: diag(W) = 1 means rows of V have unit norm
    
    # Alternative: work with W directly
    step = 0.1
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Gradient step
        W = W + step * adj
        
        # Project diagonal to 1
        np.fill_diagonal(W, 1.0)
        
        # Project to PSD: eigendecomposition
        try:
            evals, evecs = np.linalg.eigh(W)
            # Clip negative eigenvalues to 0
            evals = np.maximum(evals, 0)
            W = evecs @ np.diag(evals) @ evecs.T
            # Re-project diagonal
            np.fill_diagonal(W, 1.0)
        except:
            break
    
    # Goemans-Williamson rounding
    # W = V V^T where columns of V are the vectors
    try:
        evals, evecs = np.linalg.eigh(W)
        evals = np.maximum(evals, 0)
        # V = evecs * sqrt(evals)
        V = evecs * np.sqrt(evals)[None, :]
    except:
        return _spectral_init(adj, n)
    
    # Random hyperplane rounding
    rng = np.random.RandomState(42)
    best_labels = None
    best_cut = -1
    
    # Try multiple random hyperplanes
    for _ in range(50):
        if time.time() > deadline:
            break
        u = rng.randn(n)
        u /= np.linalg.norm(u) + 1e-12
        labels = (V @ u > 0).astype(np.float64)
        
        if np.all(labels == 0) or np.all(labels == 1):
            continue
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = _spectral_init(adj, n)
    
    return best_labels


def _greedy_refine(adj, labels, deadline):
    """
    Vectorized best-improvement local search with incremental updates.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    As = adj @ s
    gains = s * As
    
    while True:
        if time.time() > deadline:
            break
        
        max_gain = np.max(gains)
        
        if max_gain <= 0:
            break
        
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        s[v] = -s[v]
        
        As += 2 * adj[:, v] * s[v]
        gains = s * As
    
    return labels, 0.5 * (np.sum(adj) - s @ adj @ s)


def _random_init(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, n).astype(np.float64)


def _simulated_annealing_polish(adj, labels, deadline, start_temp=1.0, cool_rate=0.995, min_temp=0.01):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    current_cut = 0.5 * (np.sum(adj) - s @ adj @ s)
    best_cut = current_cut
    best_labels = labels.copy()
    
    temp = start_temp
    rng = np.random.RandomState(123)
    
    while temp > min_temp and time.time() < deadline:
        v = rng.randint(n)
        
        old_s_v = s[v]
        As_v = adj[v] @ s
        gain = old_s_v * As_v
        
        new_cut = current_cut + gain
        
        if new_cut >= current_cut or rng.random() < np.exp((new_cut - current_cut) / temp):
            labels[v] = 1 - labels[v]
            s[v] = -s[v]
            current_cut = new_cut
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        temp *= cool_rate
    
    return best_labels, best_cut


def _tabu_search(adj, labels, deadline, tabu_len=10, max_iter=5000):
    """
    Tabu search for escaping local optima.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    current_cut = 0.5 * (np.sum(adj) - s @ adj @ s)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Tabu list: (vertex, time_added)
    tabu = {}
    rng = np.random.RandomState(77)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute gains
        As = adj @ s
        gains = s * As
        
        # Remove expired tabu entries
        current_time = it
        for v in list(tabu.keys()):
            if current_time - tabu[v] >= tabu_len:
                del tabu[v]
        
        # Find best non-tabu move
        best_gain = -np.inf
        best_v = -1
        for v in range(n):
            if v in tabu:
                continue
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v == -1:
            # All moves are tabu, pick the best one anyway
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
        
        if best_gain <= 0:
            # No improving moves, try a random non-tabu move
            non_tabu = [v for v in range(n) if v not in tabu]
            if not non_tabu:
                non_tabu = list(range(n))
            best_v = rng.choice(non_tabu)
            best_gain = gains[best_v]
        
        # Make the move
        labels[best_v] = 1 - labels[best_v]
        s[best_v] = -s[best_v]
        tabu[best_v] = it
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Find a high-quality cut using SDR initialization, spectral initialization,
    greedy local search, tabu search, and simulated annealing with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    best_labels = None
    best_cut = -1
    
    # 1. SDR initialization + greedy refine
    if time.time() < deadline - 30.0:
        sdr_labels = _sdr_init(adj, n, deadline)
        refined_labels, cut_val = _greedy_refine(adj, sdr_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
    
    # 2. Spectral initialization + greedy refine
    if time.time() < deadline - 20.0:
        spec_labels = _spectral_init(adj, n)
        refined_labels, cut_val = _greedy_refine(adj, spec_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
    
    # 3. Tabu search on the best solution
    if best_labels is not None and time.time() < deadline - 10.0:
        tabu_labels, tabu_cut = _tabu_search(adj, best_labels, deadline)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
    
    # 4. Simulated annealing polish
    if best_labels is not None and time.time() < deadline - 5.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # 5. Random restarts with greedy refine + occasional SA/tabu polish
    rng = np.random.RandomState(42)
    restart_count = 0
    sa_every = 8
    
    while time.time() < deadline - 2.0:
        random_labels = _random_init(n)
        refined_labels, cut_val = _greedy_refine(adj, random_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        restart_count += 1
        
        if restart_count % sa_every == 0 and time.time() < deadline - 5.0:
            # Alternate between SA and tabu
            if restart_count % (2 * sa_every) == 0:
                tabu_labels, tabu_cut = _tabu_search(adj, best_labels, deadline)
                if tabu_cut > best_cut:
                    best_cut = tabu_cut
                    best_labels = tabu_labels.copy()
            else:
                sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
                if sa_cut > best_cut:
                    best_cut = sa_cut
                    best_labels = sa_labels.copy()
        
        if restart_count > 3000:
            break
    
    # 6. Final polish if time permits
    if best_labels is not None and time.time() < deadline - 1.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = _random_init(n)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
