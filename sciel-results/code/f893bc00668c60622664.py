import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute for evaluation
    adj_rows = A_bool

    # Features
    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Sum of Neighbor Degrees
    SND = A @ D
    
    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Core Numbers
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    remaining = set(range(N))
    # Simple core number calculation
    while remaining:
        min_deg = min(deg[i] for i in remaining)
        to_remove = [i for i in remaining if deg[i] == min_deg]
        if not to_remove:
            break
        for i in to_remove:
            core[i] = min_deg
            remaining.remove(i)
            for j in nbrs[i]:
                if j in remaining:
                    deg[j] -= 1

    # Fiedler Vector
    # Laplacian L = D - A
    L = np.diag(D) - A
    # Use sparse for efficiency if N is large, but N=200 is small
    # Eigenvalues of L
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is eigenvector corresponding to 2nd smallest eigenvalue
        # eigvals are sorted in ascending order
        fiedler = np.abs(eigvecs[:, 1])
    except Exception:
        fiedler = np.zeros(N)

    # Normalize features
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    SND_n = norm(SND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    Core_n = norm(core)
    Fiedler_n = norm(fiedler)

    # Fast evaluation function
    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[adj_rows[v]] = True
        return count

    # Baseline: Low Degree
    scores_base = -D_n
    best_size = evaluate_fast(scores_base)
    best_scores = scores_base.copy()

    # Check Core Number baseline
    scores_core = -Core_n
    sz_core = evaluate_fast(scores_core)
    if sz_core > best_size:
        best_size = sz_core
        best_scores = scores_core.copy()

    # Check Fiedler baseline (pick peripheral vertices)
    scores_fiedler = Fiedler_n
    sz_fiedler = evaluate_fast(scores_fiedler)
    if sz_fiedler > best_size:
        best_size = sz_fiedler
        best_scores = scores_fiedler.copy()

    # Mixed baselines
    for w_d in [0.5, 1.0, 1.5]:
        for w_c in [0.5, 1.0, 1.5]:
            s = -w_d * D_n - w_c * Core_n
            sz = evaluate_fast(s)
            if sz > best_size:
                best_size = sz
                best_scores = s.copy()

    # Random Search
    rng = np.random.default_rng(42)
    
    # Weights for: -D, -SND, -Tri, -Clus, -Core, +Fiedler
    # Note: Fiedler is added because high magnitude = peripheral, which is good for MIS
    # D, SND, Tri, Clus, Core are subtracted because high centrality is bad for MIS
    
    max_trials = 50000
    trial = 0
    while trial < max_trials:
        if time.time() > DEADLINE - 10.0:
            break

        w0 = rng.uniform(0.5, 2.0)  # D
        w1 = rng.uniform(0.0, 1.5)  # SND
        w2 = rng.uniform(0.0, 1.0)  # Tri
        w3 = rng.uniform(0.0, 1.0)  # Clus
        w4 = rng.uniform(0.5, 2.0)  # Core
        w5 = rng.uniform(0.0, 1.5)  # Fiedler

        s = (-w0*D_n - w1*SND_n - w2*Tri_n - w3*Clus_n - w4*Core_n + w5*Fiedler_n)
        
        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

        trial += 1

    # Local Refinement: Swap-based local search
    if time.time() < DEADLINE - 5.0:
        # We will try to improve the ordering by swapping vertices
        # We represent the order by the scores.
        # If we swap u and v, we need to adjust their scores.
        # A simpler approach: just perturb scores and keep if better.
        
        # More effective: identify close pairs in score and swap them.
        # Get current order
        order = np.argsort(-best_scores, kind='stable')
        
        improvements = 0
        for _ in range(1000):
            if time.time() > DEADLINE - 2.0:
                break
            if improvements == 0:
                break
            
            improvements = 0
            # Try random swaps
            for _ in range(50):
                if time.time() > DEADLINE - 2.0:
                    break
                u_idx, v_idx = rng.integers(0, N, 2)
                if u_idx == v_idx:
                    continue
                
                u, v = order[u_idx], order[v_idx]
                
                # If u is before v in order, scores[u] >= scores[v]
                # To swap, we need scores'[u] < scores'[v]
                # Let's just swap their score values
                new_scores = best_scores.copy()
                temp = new_scores[u]
                new_scores[u] = new_scores[v]
                new_scores[v] = temp
                
                # Add small noise to break ties
                new_scores[u] += rng.normal(0, 1e-6)
                new_scores[v] += rng.normal(0, 1e-6)
                
                sz = evaluate_fast(new_scores)
                if sz > best_size:
                    best_size = sz
                    best_scores = new_scores
                    # Update order to reflect new scores
                    order = np.argsort(-best_scores, kind='stable')
                    improvements += 1
                    break # Restart inner loop if improvement found

    return best_scores
