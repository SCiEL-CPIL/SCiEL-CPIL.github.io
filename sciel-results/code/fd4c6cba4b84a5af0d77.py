"""Guided local search penalty rule for the TSP (combined approach)."""
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

    # Robust reference for the current tour: median of the used edges' true lengths.
    if used.size:
        nz = used > 0.0
        if nz.any():
            ref = float(np.median(D[nz]))
        else:
            ref = float(np.median(D))
    else:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # Guided local search perturbation using exponential saturation:
    #   penalty = max_penalty * (1 - exp(-k * used / ref))
    # This gives a smooth, bounded penalty that is gentle for lightly-used edges
    # and approaches max_penalty for heavily-used edges.
    k = 1.0
    max_penalty = 1.0
    scaled = used / ref
    penalty = max_penalty * (1.0 - np.exp(-k * scaled))
    penalty = np.maximum(penalty, 0.0)

    # Create a mask for edges in the current tour to focus the penalty
    tour_mask = np.zeros_like(D, dtype=bool)
    tour_len = len(local_opt_tour)
    for i in range(tour_len):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % tour_len]
        tour_mask[u, v] = True
        tour_mask[v, u] = True

    # Apply penalty only to tour edges
    penalty = np.where(tour_mask, penalty, 0.0)

    out = D * (1.0 + penalty)

    # Enforce exact symmetry (defensive; both inputs are symmetric).
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
