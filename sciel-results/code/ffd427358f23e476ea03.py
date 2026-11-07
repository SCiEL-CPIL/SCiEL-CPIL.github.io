"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Hard wall-clock deadline (well inside the 330 s evaluator kill limit)
_DEADLINE_S = 300.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.monotonic()
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
    if tour.ndim == 1 and len(tour) >= 2:
        # Extract tour edge lengths
        tour_edges = []
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            if 0 <= a < n and 0 <= b < n:
                tour_edges.append(D[a, b])
        # Close the tour
        a, b = tour[-1], tour[0]
        if 0 <= a < n and 0 <= b < n:
            tour_edges.append(D[a, b])
        if tour_edges:
            ref = float(np.median(tour_edges))
        else:
            ref = float(np.median(D))
    else:
        ref = float(np.median(D))

    if not np.isfinite(ref) or ref <= 0.0:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # Smooth saturating penalty: alpha * x / (1 + x) where x = used / ref
    # This is bounded by alpha, smooth, and grows sub-linearly with use count.
    alpha = 1.2
    x = used / ref
    penalty = alpha * x / (1.0 + x)
    penalty = np.minimum(penalty, 1.0)  # hard cap
    penalty = np.maximum(penalty, 0.0)

    out = D * (1.0 + penalty)

    # Enforce exact symmetry (defensive; both inputs are symmetric).
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
