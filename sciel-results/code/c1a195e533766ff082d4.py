import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Combines:
    - Focused penalty on current tour edges (inspiration 2)
    - Usage-aware scaling with soft saturation (inspiration 1)
    - Small uniform penalty to encourage exploration (inspiration 2)
    - Smooth, moderate penalty magnitudes scaled by median distance
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
    
    # Reference scale: median of non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (D > 1e-10) & mask
    if np.any(nonzero_mask):
        ref = float(np.median(D[nonzero_mask]))
    else:
        ref = 1.0
    
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    
    # Parse tour edges
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
    
    # Start with zero penalty matrix
    penalty = np.zeros_like(D)
    
    # Apply usage-aware penalty only to tour edges
    # Use soft-saturating scaling: u / (1 + u) prevents explosion
    # Base penalty is moderate: 0.3 * ref
    tour_penalty_base = 0.3 * ref
    if tour_edges:
        for (a, b) in tour_edges:
            u = 0.5 * (used[a, b] + used[b, a])
            # Soft saturation: grows linearly at first, then saturates
            scaled_usage = u / (1.0 + u)
            pen = tour_penalty_base * scaled_usage
            penalty[a, b] += pen
            penalty[b, a] += pen
    
    # Add a small uniform penalty to all non-diagonal elements
    # This encourages exploration without heavily distorting the metric
    small_uniform = 0.015 * ref
    penalty += small_uniform
    np.fill_diagonal(penalty, 0.0)
    
    # Hard cap to prevent over-penalization
    max_penalty = 1.0 * ref
    penalty = np.minimum(penalty, max_penalty)
    penalty = np.maximum(penalty, 0.0)
    
    # Add penalty to the distance matrix
    out = D + penalty
    
    # Enforce exact symmetry
    out = 0.5 * (out + out.T)
    
    # Ensure diagonal is zero
    np.fill_diagonal(out, 0.0)
    
    # Final finiteness check
    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)
        np.fill_diagonal(out, 0.0)
    
    return out
