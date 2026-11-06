# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1  # ±1
    # s^T A s counts 2 for each same-side edge and -2 for each crossing edge? No.
    # Let's use the direct formula:
    # cut = sum_{i,j} A[i,j] * (labels[i] != labels[j])
    # In terms of s: labels[i] != labels[j] iff s[i]*s[j] == -1
    # (1 - s[i]*s[j])/2 is 1 if different, 0 if same
    # cut = 0.5 * sum_{i,j} A[i,j] * (1 - s[i]*s[j])
    #     = 0.5 * (sum A - s^T A s)
    # But sum A is 2*E (since undirected, A is symmetric with 0 diagonal)
    # Actually, let's just compute it directly to be safe and fast with numpy
    # For 100 nodes, O(n^2) is 10,000 ops, which is trivial.
    
    # Efficient vectorized way:
    # diff = labels[:, None] != labels[None, :]
    # return np.sum(adj * diff)
    
    # Even faster using the s formulation:
    # s^T A s = sum_i sum_j s_i A_ij s_j
    # For undirected graph with A symmetric:
    # s^T A s = sum_{i<j} 2 * A_ij * s_i s_j
    # If s_i s_j = 1 (same), contributes 2 A_ij
    # If s_i s_j = -1 (diff), contributes -2 A_ij
    # Let S = sum of same-side edges (counted once), D = sum of diff-side edges (counted once)
    # s^T A s = 2S - 2D
    # Also 2E = 2S + 2D => S + D = E
    # So 2S - 2D = 2S - 2(E - S) = 4S - 2E
    # => S = (s^T A s + 2E) / 4
    # D = E - S = E - (s^T A s + 2E)/4 = (4E - s^T A s - 2E)/4 = (2E - s^T A s)/4
    # Cut size is D.
    
    # Let's just do the direct boolean mask approach, it's clear and fast enough for n=100
    diff = (labels[:, None] != labels[None, :])
    return np.sum(adj * diff)


def _spectral_init(adj, n):
    """Generate initial labels using spectral method (Fiedler vector)."""
    # Build Laplacian L = D - A
    # For undirected graph
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    try:
        evals, evecs = np.linalg.eigh(L)
        # evecs[:, 1] is the Fiedler vector
        fiedler = evecs[:, 1]
    except:
        # Fallback to random
        return np.random.randint(0, 2, n).astype(np.float64)
    
    # Use sign of Fiedler vector as initial labels
    labels = (fiedler > 0).astype(np.float64)
    
    # Ensure both classes are non-empty
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n).astype(np.float64)
    
    return labels


def _sdr_init(adj, n, deadline, n_iter=2000, seed=42):
    """
    Generate initial labels using Semidefinite Relaxation (SDR) with
    Goemans-Williamson randomized rounding.
    
    Solves the SDP relaxation of Max-Cut via projected gradient ascent,
    then uses random hyperplane rounding to extract binary labels.
    """
    rng = np.random.RandomState(seed)
    
    # Max-Cut SDP relaxation:
    # max  (1/4) * sum_{i,j} A_ij (1 - V_i . V_j)
    # s.t. V_i . V_i = 1 for all i
    #     V is n x n with unit norm columns
    #     (implicitly symmetric PSD)
    #
    # Equivalent: max (1/4) * (2*E - sum_{i,j} A_ij * V_i . V_j)
    #           = (1/2)*E - (1/4) * sum_{i,j} A_ij * V_i . V_j
    #
    # Let X = V^T V (n x n Gram matrix). Then:
    # max (1/4) * sum_{i,j} A_ij (1 - X_ij)
    # s.t. X_ii = 1, X PSD
    #
    # Gradient with respect to X: -(1/4) * A
    # But we optimize over V directly:
    # f(V) = (1/4) * sum_{i,j} A_ij (1 - V_i . V_j)
    #      = (1/4) * (2E - trace(A * V^T V))
    #      = (1/2)*E - (1/4) * trace(A * V^T V)
    #
    # df/dV_ij = -(1/4) * sum_k A_jk * V_ki ... let me think more carefully.
    #
    # f(V) = (1/4) * (2E - sum_{i,j} A_ij * dot(V_i, V_j))
    # Let's define B = A (the adjacency matrix).
    # g(V) = -sum_{i,j} B_ij * V_i^T V_j = -trace(V^T B V) = -sum_i V_i^T (B V)_i
    # dg/dV_i = -2 B V_i  (since B is symmetric)
    # df/dV_i = -(1/4) * 2 B V_i = -(1/2) B V_i
    #
    # So gradient of f w.r.t. V is: -(1/2) * B @ V
    # We want to MAXIMIZE f, so we do gradient ASCENT:
    # V_new = V + alpha * (1/2) * B @ V
    # Then project each column back to unit norm.
    
    E = 0.5 * np.sum(adj)
    B = adj  # adjacency matrix (symmetric, 0 diagonal)
    
    # Initialize V with random unit vectors
    V = rng.randn(n, n)
    norms = np.linalg.norm(V, axis=0, keepdims=True)
    V = V / np.maximum(norms, 1e-10)
    
    # Projected gradient ascent
    best_V = V.copy()
    best_obj = -np.inf
    
    alpha = 0.5  # step size
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Gradient of f(V) = (1/4)(2E - trace(V^T B V))
        # df/dV = -(1/2) B V
        # For ascent: V += alpha * (1/2) B V
        grad = (1.0 / 2.0) * (B @ V)
        
        V = V + alpha * grad
        
        # Project: normalize each column to unit norm
        norms = np.linalg.norm(V, axis=0, keepdims=True)
        V = V / np.maximum(norms, 1e-10)
        
        # Compute objective
        X = V.T @ V  # Gram matrix
        obj = 0.25 * (2 * E - np.sum(adj * X))
        
        if obj > best_obj:
            best_obj = obj
            best_V = V.copy()
        
        # Adaptive step size: reduce if objective decreased
        if it > 0:
            # Simple: keep alpha constant, it's a relaxation anyway
            pass
    
    # Goemans-Williamson randomized rounding:
    # Pick random unit vector u, label i = 0 if u . V_i >= 0, else 1
    # Do multiple random hyperplanes and keep best
    labels_list = []
    n_rounds = 20
    
    for r in range(n_rounds):
        if time.time() > deadline:
            break
        u = rng.randn(n)
        u = u / np.linalg.norm(u)
        
        # labels[i] = 0 if u . V[:, i] >= 0, else 1
        projections = u @ best_V  # shape (n,)
        labels = (projections < 0).astype(np.float64)
        
        if np.all(labels == 0) or np.all(labels == 1):
            # Degenerate, skip
            continue
        
        labels_list.append(labels)
    
    if not labels_list:
        # Fallback: use sign of first column
        labels = (best_V[:, 0] > 0).astype(np.float64)
        if np.all(labels == 0) or np.all(labels == 1):
            labels = rng.randint(0, 2, n).astype(np.float64)
        return labels
    
    # Pick the one with best cut
    best_labels = None
    best_cut = -1
    for lab in labels_list:
        diff = (lab[:, None] != lab[None, :])
        cut = np.sum(adj * diff)
        if cut > best_cut:
            best_cut = cut
            best_labels = lab
    
    return best_labels


def _greedy_refine(adj, labels, deadline):
    """
    Vectorized best-improvement local search with incremental updates.
    Repeatedly find the vertex whose flip increases the cut the most.
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
        
        # Update As incrementally: As[u] changes by -2*A[u,v]*old_s_v = +2*A[u,v]*new_s_v
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


def partition(adjacency):
    """
    Find a high-quality cut using SDR-based initialization, spectral initialization,
    greedy local search, and simulated annealing polish with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # 1. SDR initialization + greedy refine (primary method)
    sdr_deadline = min(deadline, time.time() + 30.0)
    sdr_labels = _sdr_init(adj, n, sdr_deadline)
    refined_labels, cut_val = _greedy_refine(adj, sdr_labels, deadline)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # 2. Spectral initialization + greedy refine
    spec_labels = _spectral_init(adj, n)
    refined_labels, cut_val = _greedy_refine(adj, spec_labels, deadline)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # 3. Simulated annealing polish on the best solution
    if time.time() < deadline - 5.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # 4. Random restarts with greedy refine + occasional SA polish
    rng = np.random.RandomState(42)
    restart_count = 0
    sa_every = 10
    
    while time.time() < deadline - 2.0:
        random_labels = _random_init(n)
        refined_labels, cut_val = _greedy_refine(adj, random_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        restart_count += 1
        
        if restart_count % sa_every == 0 and time.time() < deadline - 5.0:
            sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        if restart_count > 2000:
            break
    
    # 5. Final SA polish if time permits
    if time.time() < deadline - 1.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = _random_init(n)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
