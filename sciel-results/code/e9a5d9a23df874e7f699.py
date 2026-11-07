"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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

    # Reference scale: median of edge lengths of edges that have been used.
    if used.size:
        nz = used > 0.0
        if nz.any():
            ref = float(np.median(D[nz]))
        else:
            ref = float(np.median(D))
    else:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # Penalty parameters:
    #   penalty = A * (1 - exp(-k * used / ref))
    # Stronger penalty for edges in the current local-optimum tour to break
    # the local optimum more aggressively.
    k = 1.5
    A = 2.0
    scaled = used / ref
    penalty = A * (1.0 - np.exp(-k * scaled))
    penalty = np.clip(penalty, 0.0, A)

    # Extra penalty for edges in the current local-optimum tour:
    # these are the edges we most want to break to escape the local optimum.
    tour_penalty = np.zeros_like(D)
    m = len(local_opt_tour)
    for i in range(m):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % m]
        tour_penalty[a, b] += 1.0
        tour_penalty[b, a] += 1.0

    # Scale the tour penalty: edges in the tour get an extra boost
    tour_boost = 0.8 * (1.0 - np.exp(-k * tour_penalty / max(ref, 1e-9)))
    penalty = penalty + tour_boost
    penalty = np.clip(penalty, 0.0, A + 0.8)

    # Multiply the true distances by (1 + penalty) for used edges.
    out = D * (1.0 + penalty)

    # Enforce exact symmetry.
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
