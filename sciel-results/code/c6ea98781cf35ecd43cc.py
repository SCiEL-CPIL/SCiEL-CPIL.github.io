"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np

# Hard wall-clock deadline (well inside the 330 s evaluator kill limit)
_DEADLINE_S = 300.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        import time
        _START = time.monotonic()
    import time
    return time.monotonic() - _START


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: if we ever approach the limit, return the safe base matrix.
    if _elapsed() > _DEADLINE_S - 30.0:
        return np.asarray(edge_distance, dtype=float).copy()

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()

    n = D.shape[0]

    # Guard against non-finite input; fall back to the (clean) base matrix.
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base

    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)

    # Reference scale: median of edge lengths in the current local-optimal tour.
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.ndim == 1 and tour.size >= 2:
        m = tour.size
        idx_i = tour
        idx_j = np.roll(tour, -1)
        tour_edges = D[idx_i, idx_j]
        finite_tour = tour_edges[np.isfinite(tour_edges)]
        if finite_tour.size > 0:
            ref = float(np.median(finite_tour))
        else:
            ref = float(np.median(D))
    else:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # Guided local search perturbation:
    #   penalize every edge of the local optimum (and a bit of its recent history)
    #   proportionally to its use count, with a fixed, capped maximum penalty.
    penalty = 0.5 * (used / ref)
    penalty = np.minimum(penalty, 0.5)          # cap the per-edge multiplier boost
    penalty = np.maximum(penalty, 0.0)

    out = D * (1.0 + penalty)

    # Enforce exact symmetry (defensive; both inputs are symmetric).
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
