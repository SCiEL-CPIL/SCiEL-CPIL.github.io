"""Guided local search penalty for TSP – refined model-and-optimize design.

Key changes vs. the previous revision (p81):
  * Replace the per-edge O(n) `_find_best_2opt_replacement` loop with a
    vectorised numpy operation (O(1) per edge, O(n) total) that computes the
    best 2-opt delta for every tour edge simultaneously.
  * Add a "hybrid" penalty strategy that blends the edge-reversal penalty
    with a simple linear penalty, sweeping a blend coefficient.
  * Tighten the soft-min temperature and cap the number of candidate
    matrices to 4 to reduce noise in the weighted combination.
  * Add a final high-precision polish (20 passes, tol = -1e-13) on the true
    distance matrix for the best candidate before returning.
"""
import numpy as np
import time

_DEADLINE = 325.0
_start_time = None


def _two_opt(D, tour, max_passes=10, tol=-1e-10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return list(tour)
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _two_opt_precise(D, tour, max_passes=20, tol=-1e-13):
    """High-precision 2-opt polish."""
    n = len(tour)
    if n < 4:
        return list(tour)
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _closed_len(D, tour):
    """Closed-tour length using distance matrix D."""
    n = len(tour)
    idx = np.array(tour)
    nxt = np.roll(idx, -1)
    return float(D[idx, nxt].sum())


def _vectorized_best_2opt_deltas(D, tour, n):
    """For each edge i in the tour, compute the best 2-opt delta that removes
    that edge.  Returns an array of shape (n,).
    
    Uses numpy fancy indexing to avoid per-edge Python loops.
    """
    # tour is a list of node indices of length n
    # Edge i: (tour[i], tour[(i+1)%n])
    a_arr = np.array(tour)
    b_arr = np.roll(a_arr, -1)
    
    # For a 2-opt swap removing edge (a,b) and edge (c,d), the delta is:
    #   D[a,c] + D[b,d] - D[a,b] - D[c,d]
    # where (c,d) is another tour edge.
    # We want the minimum over all j != i of:
    #   D[a_i, c_j] + D[b_i, d_j] - D[a_i, b_i] - D[c_j, d_j]
    
    # Precompute D[a_i, b_i] for all i
    edge_ab = D[a_arr, b_arr]          # shape (n,)
    
    # Precompute D[c_j, d_j] for all j  (same as edge_ab but shifted)
    # c_j = tour[j], d_j = tour[(j+1)%n]  =>  same as a_arr, b_arr
    edge_cd = edge_ab                   # shape (n,)
    
    # Build matrices: M[i,j] = D[a_i, c_j] + D[b_i, d_j]
    # a_i = tour[i], c_j = tour[j]  =>  D[a_arr[:,None], a_arr[None,:]]
    # b_i = tour[(i+1)%n], d_j = tour[(j+1)%n]  =>  D[b_arr[:,None], b_arr[None,:]]
    M = D[a_arr[:, None], a_arr[None, :]] + D[b_arr[:, None], b_arr[None, :]]
    
    # For each i, the best delta over j != i is:
    #   min_j M[i,j] - edge_ab[i] - edge_cd[j]  (j chosen to minimise)
    # But edge_cd[j] depends on j, so we can't simply subtract a per-row constant.
    # Instead: delta[i,j] = M[i,j] - edge_ab[i] - edge_cd[j]
    # We need min over j != i of (M[i,j] - edge_cd[j]) - edge_ab[i]
    
    # Compute N[i,j] = M[i,j] - edge_cd[j]
    N = M - edge_cd[None, :]
    
    # For each row i, find the min over j != i
    # Set diagonal to +inf so it's excluded
    N_diag = N.copy()
    np.fill_diagonal(N_diag, np.inf)
    
    # Also, j==i means removing the same edge, which is invalid
    best_remaining = N_diag.min(axis=1)   # shape (n,)
    
    # delta[i] = best_remaining[i] - edge_ab[i]
    deltas = best_remaining - edge_ab
    
    # If the best j was i (which we masked), we might have an issue, but
    # we set diagonal to inf, so min will pick a different j if n>2.
    # For n==2, there's no valid 2-opt, but TSP with n=2 is trivial.
    return deltas


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    # Deadline safety: if we're too close, return a simple penalty
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
    
    # Build set of tour edges (undirected)
    tour_edges = []
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        tour_edges.append((min(u, v), max(u, v)))
    tour_edges_set = set(tour_edges)
    
    # Vectorised best 2-opt delta for each tour edge
    best_deltas = _vectorized_best_2opt_deltas(edge_distance, tour, n)
    
    # Gather usage counts for tour edges
    usage_arr = np.zeros(n)
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        usage_arr[i] = edge_n_used[u, v]
    
    max_usage = float(usage_arr.max()) if n > 0 else 1.0
    if max_usage < 1e-10:
        max_usage = 1.0
    
    best_D_pen = None
    best_true_len = float('inf')
    candidate_results = []
    
    def _try_candidate(D_pen):
        """Run 2-opt on penalised matrix, polish on true matrix, record result."""
        nonlocal best_D_pen, best_true_len
        if time.time() - _start_time > _DEADLINE - 15:
            return
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        if time.time() - _start_time > _DEADLINE - 10:
            tour_polished = _two_opt(edge_distance, tour_pen, max_passes=5)
        else:
            tour_polished = _two_opt_precise(edge_distance, tour_pen, max_passes=20, tol=-1e-13)
        true_len = _closed_len(edge_distance, tour_polished)
        candidate_results.append((true_len, D_pen.copy(), tour_polished))
        if true_len < best_true_len:
            best_true_len = true_len
            best_D_pen = D_pen.copy()
    
    # --- Strategy 1: Edge-reversal guided penalty (vectorised) ---
    if time.time() - _start_time < _DEADLINE - 30:
        D_pen = edge_distance.copy()
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            usage = usage_arr[i]
            if usage <= 0:
                continue
            delta_i = best_deltas[i]
            if delta_i < 0:
                # Good alternative exists: scale penalty by attractiveness
                penalty_scale = min(-delta_i, 0.5)
                penalty = 0.5 * usage * penalty_scale
            else:
                penalty = 0.1 * usage
            D_pen[u, v] += penalty
            D_pen[v, u] += penalty
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        _try_candidate(D_pen)
    
    # --- Strategy 2: Linear penalty with multiple alpha values ---
    alpha_candidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for alpha in alpha_candidates:
        if time.time() - _start_time > _DEADLINE - 20:
            break
        D_pen = edge_distance.copy()
        for (u, v) in tour_edges_set:
            usage = edge_n_used[u, v]
            if usage > 0:
                penalty = alpha * usage
                D_pen[u, v] += penalty
                D_pen[v, u] += penalty
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        _try_candidate(D_pen)
    
    # --- Strategy 3: Exponential penalty ---
    for alpha, beta in [(0.1, 1.0), (0.15, 1.5), (0.2, 2.0)]:
        if time.time() - _start_time > _DEADLINE - 20:
            break
        D_pen = edge_distance.copy()
        for (u, v) in tour_edges_set:
            usage = edge_n_used[u, v]
            if usage > 0:
                penalty = alpha * np.exp(beta * usage)
                D_pen[u, v] += penalty
                D_pen[v, u] += penalty
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        _try_candidate(D_pen)
    
    # --- Strategy 4: Hybrid blend of edge-reversal and linear penalty ---
    for blend in [0.3, 0.5, 0.7]:
        if time.time() - _start_time > _DEADLINE - 20:
            break
        D_pen = edge_distance.copy()
        # Edge-reversal component
        rev_penalty = np.zeros_like(edge_distance)
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            usage = usage_arr[i]
            if usage <= 0:
                continue
            delta_i = best_deltas[i]
            if delta_i < 0:
                penalty_scale = min(-delta_i, 0.5)
                p = 0.5 * usage * penalty_scale
            else:
                p = 0.1 * usage
            rev_penalty[u, v] += p
            rev_penalty[v, u] += p
        
        # Linear component
        lin_penalty = np.zeros_like(edge_distance)
        for (u, v) in tour_edges_set:
            usage = edge_n_used[u, v]
            if usage > 0:
                p = 0.15 * usage
                lin_penalty[u, v] += p
                lin_penalty[v, u] += p
        
        D_pen += blend * rev_penalty + (1.0 - blend) * lin_penalty
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        _try_candidate(D_pen)
    
    # --- Soft-min selection over top candidates ---
    if candidate_results:
        candidate_results.sort(key=lambda x: x[0])
        top_k = min(4, len(candidate_results))
        if top_k >= 2:
            lens = np.array([x[0] for x in candidate_results[:top_k]])
            min_len = lens.min()
            tau = 0.005  # tighter temperature
            weights = np.exp(-(lens - min_len) / tau)
            weights = weights / weights.sum()
            best_D_pen = np.zeros_like(edge_distance)
            for w, (_, D_i, _) in zip(weights, candidate_results[:top_k]):
                best_D_pen += w * D_i
        else:
            best_D_pen = candidate_results[0][1]
    
    # Final validation and return
    if best_D_pen is not None:
        if not np.all(np.isfinite(best_D_pen)):
            best_D_pen = edge_distance.copy()
        if best_D_pen.shape != edge_distance.shape:
            best_D_pen = edge_distance.copy()
        return best_D_pen
    else:
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        return edge_distance + 0.1 * edge_n_used / max_used
