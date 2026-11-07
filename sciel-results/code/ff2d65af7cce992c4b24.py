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
    Direct approach with repulsive force field around tour edges.
    
    Key ideas:
    1. Base penalty scales with usage count (soft-saturating)
    2. Tour edges get a usage-aware boost (heavily-used tour edges penalized more)
    3. A smooth 'field' penalty decays with metric distance to nearest tour edge
    4. All penalties scaled by median edge length for adaptivity
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
    
    # Focused penalty: primarily on tour edges with usage-aware scaling
    # Very small base penalty for non-tour used edges
    base_rate = 0.05
    penalty = base_rate * ref * used / (1.0 + used)
    
    # Tour edge penalties: stronger, usage-scaled
    tour_boost_base = 0.4 * ref
    if tour_edges:
        for (a, b) in tour_edges:
            u = 0.5 * (used[a, b] + used[b, a])
            # Saturating usage scaling: more penalty for heavily used edges
            boost = tour_boost_base * (1.0 + 0.5 * u / (1.0 + u))
            penalty[a, b] += boost
            penalty[b, a] += boost
        
        # Localized field penalty: only for edges very close in distance to tour edges
        # This creates a narrow exclusion zone without over-constraining
        tour_dists = np.array([D[a, b] for (a, b) in tour_edges])
        diff = np.abs(D[:, :, np.newaxis] - tour_dists[np.newaxis, np.newaxis, :])
        min_diff = np.min(diff, axis=2)  # shape (n, n)
        
        # Tight decay: only penalize edges within ~0.1*ref of tour edge distances
        field_rate = 0.1
        decay_scale = 0.15 * ref
        field_penalty = field_rate * ref * np.exp(-min_diff / decay_scale)
        
        # Zero out diagonal and tour edges (already penalized above)
        np.fill_diagonal(field_penalty, 0.0)
        for (a, b) in tour_edges:
            field_penalty[a, b] = 0.0
            field_penalty[b, a] = 0.0
        
        penalty += field_penalty
    
    # Hard cap
    max_penalty = 0.8 * ref
    penalty = np.minimum(penalty, max_penalty)
    penalty = np.maximum(penalty, 0.0)
    
    # Add penalty to the distance matrix
    out = D + penalty
    
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
