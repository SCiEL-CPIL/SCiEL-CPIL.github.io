"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_S = 330.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.monotonic()
    return time.monotonic() - _START


def _safe_base(D):
    base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)
    return base


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: return the clean base matrix if we are near the limit.
    if _elapsed() > _DEADLINE_S - 40.0:
        return _safe_base(np.asarray(edge_distance, dtype=float))

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return _safe_base(D)

    n = D.shape[0]
    if not np.all(np.isfinite(D)):
        return _safe_base(D)

    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)

    # Reference scale from edges that have actually been used.
    nz = used > 0.0
    if nz.any():
        ref = float(np.median(D[nz]))
    else:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # ---- Decayed effective usage: recent rounds matter more than ancient ones.
    # Without decay the penalty saturates on old edges and stops providing
    # fresh pressure to escape the current basin.
    max_used = float(np.max(used)) if used.size else 0.0
    if max_used > 0.0:
        # Infer the current round (number of accumulated additions).
        # usage counts are integers added each round; the max count approx
        # round index. Use a decay factor so older usage fades.
        decay = 0.78 ** max_used
        eff = used * decay
    else:
        eff = used.copy()

    # ---- Base penalty: strong, smoothly saturating, on decayed usage.
    k = 2.0
    A = 3.0
    scaled = eff / ref
    penalty = A * (1.0 - np.exp(-k * scaled))
    penalty = np.clip(penalty, 0.0, A)

    # ---- Extra penalty on the current local-optimum tour edges.
    # These are the edges we most want to break to escape the local optimum.
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    if m >= 2:
        i = np.arange(m)
        a = tour[i]
        b = tour[(i + 1) % m]
        tour_mask = np.zeros_like(D)
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
        # Boost on top of base penalty; bounded so 2-opt stays well-behaved.
        tour_boost = 1.5 * tour_mask
        penalty = penalty + tour_boost

    penalty = np.clip(penalty, 0.0, A + 1.5)

    # Multiply true distances by (1 + penalty) for perturbed edges.
    out = D * (1.0 + penalty)

    # Enforce exact symmetry and zero diagonal.
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)

    if not np.all(np.isfinite(out)):
        return _safe_base(D)

    return out
# EVOLVE-BLOCK-END
