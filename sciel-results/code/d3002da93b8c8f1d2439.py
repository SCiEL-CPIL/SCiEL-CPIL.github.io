"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search with focused tour-edge penalties.
    
    Key ideas:
    1. Only current tour edges receive penalties (not all used edges)
    2. Penalty on tour edges scales with their usage count
    3. Moderate penalty magnitude to encourage restructuring without over-distorting
    4. All scaled by median edge length for adaptivity
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()
    
    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()
    
    n = D.shape[0]
    
    # Guard against non-finite input
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base
    
    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)
    
    # Reference scale
    ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    
    # Parse tour
    tour = np.asarray(local_opt_tour, dtype=int)
    tour_edges = []
    if tour.ndim == 1 and len(tour) >= 2:
        for i in range(len(tour) - 1):
            a, b = int(tour[i]), int(tour[i + 1])
            if 0 <= a < n and 0 <= b < n:
                tour_edges.append((a, b))
        a, b = int(tour[-1]), int(tour[0])
        if 0 <= a < n and 0 <= b < n:
            tour_edges.append((a, b))
    
    # Penalty only on current tour edges, scaled by usage
    out = D.copy()
    if tour_edges:
        # Usage-aware penalty: more used tour edges get higher penalty
        # Use a moderate base with usage scaling
        base_penalty = 0.8 * ref
        for (a, b) in tour_edges:
            u = 0.5 * (used[a, b] + used[b, a])
            # Soft-saturating: penalty grows with usage but bounded
            penalty_val = base_penalty * (1.0 + 0.5 * u / (1.0 + u))
            out[a, b] += penalty_val
            out[b, a] += penalty_val
    
    # Enforce exact symmetry
    out = 0.5 * (out + out.T)
    
    # Ensure diagonal is zero
    np.fill_diagonal(out, 0.0)
    
    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)
        np.fill_diagonal(out, 0.0)
    
    return out
# EVOLVE-BLOCK-END
