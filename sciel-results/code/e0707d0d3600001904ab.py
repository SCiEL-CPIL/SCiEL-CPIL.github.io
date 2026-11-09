# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    deadline = time.time() + 110.0  # safety margin under 120s / 150s kill

    # Work in {+1, -1} internally; convert to 0/1 at the end.
    adj = np.asarray(adjacency, dtype=np.float64)
    deg = adj.sum(axis=1)

    def cut_size(s):
        # s in {+1,-1}; crossing edges = 0.5 * sum_{i,j} A[i,j]*(1 - s[i]*s[j])
        return 0.5 * (deg.sum() - (s * (adj @ s)).sum())

    def local_improve(s):
        # Greedy flip until no improving move; vectorized gain per pass.
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # edges from each vertex to the opposite side
            opp = 1.0 - s  # 1 where opposite to +1, 0 where same as +1
            # For vertex v: edges to opposite = sum_u A[v,u] * (1 - s[v]*s[u])/2
            # = 0.5 * ( (adj @ (1 - s*s[:,None]))[v] ) ... do directly:
            # cross[v] = sum_u A[v,u] * (1 - s[v]*s[u]) / 2
            cross = 0.5 * ((adj * (1.0 - s[:, None] * s[None, :])).sum(axis=1))
            gain = 2.0 * cross - deg  # flipping v changes cut by this
            # flip any vertex with positive gain (do in one pass, recompute)
            flip = gain > 1e-9
            if flip.any():
                s[flip] = -s[flip]
                improved = True
        return s

    # --- Spectral embedding: top eigenvectors of the Laplacian ---
    L = np.diag(deg) - adj
    # Use dense eig for n=100 (fast and reliable)
    try:
        w, V = np.linalg.eigh(L)  # ascending
    except Exception:
        w, V = np.linalg.eig(L)
        idx = np.argsort(np.real(w))
        w = np.real(w[idx]); V = np.real(V[:, idx])
    # Take top k eigenvectors (largest eigenvalues) -> strongest separation
    k = min(8, n)
    top = V[:, -k:]
    top = top / (np.linalg.norm(top, axis=0, keepdims=True) + 1e-12)

    best_s = None
    best_val = -1.0

    def try_partition(s):
        nonlocal best_s, best_val
        s = local_improve(s.copy())
        v = cut_size(s)
        if v > best_val:
            best_val = v
            best_s = s.copy()

    # Random hyperplane projections of the embedding
    rng = np.random.default_rng(12345)
    n_proj = 64
    for i in range(n_proj):
        if time.time() > deadline:
            break
        r = rng.standard_normal(k)
        r /= (np.linalg.norm(r) + 1e-12)
        coords = top @ r
        s = np.where(coords >= 0, 1.0, -1.0)
        try_partition(s)

    # Also try a few pure random restarts as a safety net
    for i in range(16):
        if time.time() > deadline:
            break
        s = rng.choice([-1.0, 1.0], size=n)
        try_partition(s)

    if best_s is None:
        best_s = np.ones(n)

    # Convert {+1,-1} -> {0,1}
    labels = np.where(best_s > 0, 1, 0).astype(int)
    return labels
# EVOLVE-BLOCK-END
