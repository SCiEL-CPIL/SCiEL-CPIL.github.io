# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Compute priority scores for vertices for the fixed greedy independent-set decoder.
    Higher score => selected earlier.
    """
    start_time = time.time()
    DEADLINE = 135.0

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]

    if n == 0:
        return np.array([])

    # Precompute neighbor lists as lists of ints for fast iteration
    neighbors = [np.flatnonzero(adj[i]) for i in range(n)]
    deg = np.array([len(neighbors[i]) for i in range(n)], dtype=np.float64)

    # ---- Structural baseline (cheap) -------------------------------------
    adj_f = adj.astype(np.float64)
    neighbor_degree_sums = adj_f @ deg
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = np.where(deg > 0, neighbor_degree_sums / np.maximum(deg, 1.0), 0.0)

    # Triangle counts: number of triangles each vertex participates in
    # tri_count[i] = sum over neighbors j of (number of common neighbors of i and j)
    # = 0.5 * sum over neighbors j of (A @ A)[i,j]
    try:
        A2 = adj_f @ adj_f
        tri_counts = np.array([np.sum(A2[i, neighbors[i]]) for i in range(n)], dtype=np.float64)
    except Exception:
        tri_counts = np.zeros(n, dtype=np.float64)

    # Structural score: reward low degree, high neighbor degree, low triangle count
    struct = (mean_neighbor_degrees - deg) + 0.5 / (1.0 + deg) - 0.1 * tri_counts / (tri_counts.max() + 1e-9)

    # Spectral feature: Fiedler vector
    fiedler_mag = np.zeros(n, dtype=np.float64)
    if n >= 3:
        try:
            if time.time() - start_time < DEADLINE - 10.0:
                deg_int = deg.astype(np.int64)
                L = csr_matrix((deg_int - adj_f), shape=(n, n))
                try:
                    vals, vecs = eigsh(L, k=2, sigma=-1.0, which='LM')
                    idx = np.argsort(vals)
                    if len(idx) >= 2:
                        fiedler_vec = vecs[:, idx[1]]
                        fiedler_mag = np.abs(fiedler_vec)
                except Exception:
                    try:
                        vals, vecs = eigsh(L, k=2, which='SM')
                        idx = np.argsort(vals)
                        if len(idx) >= 2:
                            fiedler_vec = vecs[:, idx[1]]
                            fiedler_mag = np.abs(fiedler_vec)
                    except Exception:
                        pass
        except Exception:
            pass

    if time.time() - start_time > DEADLINE:
        return struct

    # ---- Greedy decoder function ------------------------------------------
    def greedy_size(order):
        """Return the size of the independent set produced by the greedy decoder."""
        remaining = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if not remaining[v]:
                continue
            count += 1
            remaining[v] = False
            if len(neighbors[v]) > 0:
                remaining[neighbors[v]] = False
        return count

    def greedy_set(order):
        """Return the set of selected vertices."""
        remaining = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        for v in order:
            if not remaining[v]:
                continue
            selected[v] = True
            remaining[v] = False
            if len(neighbors[v]) > 0:
                remaining[neighbors[v]] = False
        return selected

    # ---- Local search over priority scores ---------------------------------
    # Start from structural score, then do pairwise swaps to improve greedy size
    scores = struct.copy()
    order = np.argsort(-scores)
    best_size = greedy_size(order)

    rng = np.random.default_rng(42)

    # Trial to estimate cost
    t0 = time.time()
    greedy_size(order)
    per_eval = max(time.time() - t0, 1e-6)

    remaining_time = DEADLINE - (time.time() - start_time)
    # Each swap evaluation is one greedy_size call
    max_swaps = int(max(10, remaining_time / (per_eval * 1.2)))
    max_swaps = min(max_swaps, 5000)

    # Local search: try random pairwise swaps, keep if improvement
    for it in range(max_swaps):
        if time.time() - start_time > DEADLINE:
            break

        # Pick two random positions to swap
        i = rng.integers(0, n - 1)
        j = rng.integers(i + 1, n)

        # Swap in order
        order[i], order[j] = order[j], order[i]
        new_size = greedy_size(order)

        if new_size > best_size:
            best_size = new_size
            # Update scores to reflect new order
            # Assign scores based on position in order (earlier = higher)
            scores = np.zeros(n, dtype=np.float64)
            scores[order] = np.arange(n, 0, -1, dtype=np.float64)
        else:
            # Revert
            order[i], order[j] = order[j], order[i]

    # ---- Final blend: use the locally-searched ordering as primary signal ----
    # The locally-searched order directly optimizes the metric
    # Assign scores from the order, blended with structural for robustness
    order_scores = np.zeros(n, dtype=np.float64)
    order_scores[order] = np.arange(n, 0, -1, dtype=np.float64)

    # Normalize both to zero mean, unit variance
    ostd = order_scores.std()
    sstd = struct.std()
    fstd = fiedler_mag.std()

    if ostd > 1e-12 and sstd > 1e-12:
        order_norm = (order_scores - order_scores.mean()) / ostd
        struct_norm = (struct - struct.mean()) / sstd
        if fstd > 1e-12:
            fiedler_norm = (fiedler_mag - fiedler_mag.mean()) / fstd
            # Heavily weight the locally-optimized order
            final_scores = 0.70 * order_norm + 0.15 * struct_norm + 0.15 * fiedler_norm
        else:
            final_scores = 0.80 * order_norm + 0.20 * struct_norm
    else:
        final_scores = order_scores

    return final_scores
# EVOLVE-BLOCK-END
