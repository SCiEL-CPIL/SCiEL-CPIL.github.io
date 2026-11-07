import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.

    Returns a perturbed distance matrix that biases 2-opt away from
    frequently-used edges and, most strongly, away from the edges of the
    current local-optimum tour (the ones 2-opt tends to keep).

    Penalty design:
      - Scale is the median edge distance, so the perturbation is meaningful
        regardless of the absolute size of the true distances.
      - Cumulative penalty grows with how often an edge has been used, so
        edges that survive many local-optimum tours are pushed out hardest.
      - Current-tour edges get the largest multiplier to force escape.
    """
    start_time = time.time()
    n = edge_distance.shape[0]

    # Safety: return a copy of the true distances if we cannot proceed.
    def _safe():
        out = edge_distance.copy()
        if not np.all(np.isfinite(out)):
            out = np.where(np.isfinite(out), out, 1.0)
        return out

    if n == 0:
        return _safe()

    d_out = edge_distance.copy()

    # Reference scale: median of the finite off-diagonal distances.
    finite = d_out[np.isfinite(d_out)]
    finite = finite[finite > 0]
    if finite.size == 0:
        scale = 1.0
    else:
        scale = float(np.median(finite))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0

    # Mask of edges in the current local-optimum tour.
    tour_mask = np.zeros((n, n), dtype=float)
    try:
        for i in range(n):
            u = int(local_opt_tour[i])
            v = int(local_opt_tour[(i + 1) % n])
            if 0 <= u < n and 0 <= v < n and u != v:
                tour_mask[u, v] = 1.0
                tour_mask[v, u] = 1.0
    except Exception:
        # Malformed tour: fall back to a pure usage-based penalty.
        tour_mask = np.zeros((n, n), dtype=float)

    # Cumulative usage penalty. edge_n_used is symmetric; counts grow over
    # the 20 rounds. Use a sub-linear-but-growing function so heavily used
    # edges are penalised more than lightly used ones, without exploding.
    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != (n, n):
        used = np.zeros((n, n), dtype=float)
    used = np.where(np.isfinite(used), used, 0.0)
    used = np.maximum(used, 0.0)

    # Penalty magnitudes (in units of the median edge distance).
    # Current-tour edges: strong, to force 2-opt to break the local optimum.
    w_cur = 0.55
    # Cumulative edges: moderate, growing with usage.
    w_cum = 0.10
    # The cumulative term grows as sqrt(used): used=1 -> 0.10, used=4 -> 0.20,
    # used=9 -> 0.30, used=16 -> 0.40. Bounded and smooth.
    cum_term = w_cum * np.sqrt(used)

    # Current-tour term.
    cur_term = w_cur * tour_mask

    # Total additive penalty in distance units.
    penalty = (cum_term + cur_term) * scale

    # Apply additively (equivalent to multiplicative but numerically clean).
    d_out = d_out + penalty

    # Keep diagonal zero and matrix symmetric.
    np.fill_diagonal(d_out, 0.0)
    d_out = 0.5 * (d_out + d_out.T)

    # Ensure finiteness.
    if not np.all(np.isfinite(d_out)):
        return _safe()

    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # return the safe (true) matrix rather than risk a timeout mid-evaluation.
    if (time.time() - start_time) > 320.0:
        return _safe()

    return d_out
