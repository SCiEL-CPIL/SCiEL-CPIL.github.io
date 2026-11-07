import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP using a repulsive force field.
    
    Creates a continuous penalty field around the current tour, where
    the penalty magnitude decays with proximity to tour edges. This
    gently discourages 2-opt from accepting swaps that bring the tour
    closer to its previous structure, encouraging exploration of new
    regions of the solution space.
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
    
    # Parse tour edges and collect tour vertices
    tour = np.asarray(local_opt_tour, dtype=int)
    tour_vertices = set()
    tour_edges = []
    if tour.ndim == 1 and len(tour) >= 2:
        for i in range(len(tour) - 1):
            a, b = int(tour[i]), int(tour[i + 1])
            if 0 <= a < n and 0 <= b < n:
                tour_edges.append((a, b))
                tour_vertices.add(a)
                tour_vertices.add(b)
        a, b = int(tour[-1]), int(tour[0])
        if 0 <= a < n and 0 <= b < n:
            tour_edges.append((a, b))
            tour_vertices.add(a)
            tour_vertices.add(b)
    
    if not tour_edges:
        return D.copy()
    
    # Compute minimum distance from each vertex to any tour vertex
    # This creates a "proximity field" around the tour
    tv_list = list(tour_vertices)
    if not tv_list:
        return D.copy()
    
    # For each vertex, find min distance to any tour vertex
    # Vectorized: D[:, tv_list] gives distances from all vertices to tour vertices
    min_dist_to_tour = np.min(D[:, tv_list], axis=1)
    
    # Create the repulsive force field penalty
    # Penalty for edge (i,j) is based on how close i and j are to the tour
    # Use the minimum of the two vertices' distances to create a "shadow"
    # that extends beyond the tour edges
    vertex_proximity = np.minimum(min_dist_to_tour[:, np.newaxis], 
                                   min_dist_to_tour[np.newaxis, :])
    
    # Apply exponential decay: edges close to tour get higher penalty
    # Decay length: ~0.3 * ref means edges within ~0.3*ref of tour are affected
    decay_length = 0.3 * ref
    if decay_length <= 0:
        decay_length = ref
    
    # Exponential decay field
    proximity_field = np.exp(-vertex_proximity / decay_length)
    
    # Base penalty strength, scaled by usage of nearby edges
    # Compute average usage of tour edges to modulate penalty strength
    tour_usage = 0.0
    for (a, b) in tour_edges:
        tour_usage += 0.5 * (used[a, b] + used[b, a])
    avg_usage = tour_usage / len(tour_edges) if tour_edges else 0.0
    
    # Soft-saturate usage to prevent explosion
    scaled_usage = avg_usage / (1.0 + avg_usage)
    
    # Base penalty: moderate, scaled by usage
    base_penalty = 0.25 * ref * (0.5 + 0.5 * scaled_usage)
    
    # Apply the field: stronger penalty where proximity is high (close to tour)
    penalty = base_penalty * proximity_field
    
    # Add extra penalty directly on tour edges (stronger core)
    for (a, b) in tour_edges:
        u = 0.5 * (used[a, b] + used[b, a])
        scaled_u = u / (1.0 + u)
        core_pen = 0.3 * ref * scaled_u
        penalty[a, b] += core_pen
        penalty[b, a] += core_pen
    
    # Small uniform penalty for exploration
    penalty += 0.01 * ref
    
    # Zero out diagonal
    np.fill_diagonal(penalty, 0.0)
    
    # Hard cap
    max_penalty = 0.8 * ref
    penalty = np.minimum(penalty, max_penalty)
    penalty = np.maximum(penalty, 0.0)
    
    # Add penalty to distance matrix
    out = D + penalty
    
    # Enforce symmetry
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    
    # Final finiteness check
    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)
        np.fill_diagonal(out, 0.0)
    
    return out
