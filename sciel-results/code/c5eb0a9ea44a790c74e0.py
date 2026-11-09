"""Guided local search penalty for TSP - edge-reversal heuristic approach."""
import numpy as np
import time

_DEADLINE = 330.0
_start_time = None


def _two_opt(D, tour, max_passes=10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _closed_len(D, tour):
    """Compute the length of the closed tour using distance matrix D."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
    return total


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Fallback: mild uniform penalty
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        if max_used < 1e-10:
            max_used = 1.0
        penalty = 0.1 * edge_n_used / max_used
        D_out = edge_distance + penalty
        D_out = (D_out + D_out.T) / 2.0
        np.fill_diagonal(D_out, 0.0)
        return D_out

    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    # Baseline: average edge distance in the matrix (off-diagonal)
    mask_off = ~np.eye(n, dtype=bool)
    avg_edge_dist = edge_distance[mask_off].mean()
    
    # For each tour edge, find the best 2-opt swap that removes it
    # Tour edge i: (tour[i], tour[(i+1)%n])
    # A 2-opt swap removing edge i and edge j:
    #   removes (a,b) and (c,d), adds (a,c) and (b,d)
    #   delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]
    # We want min delta over all valid j (j != i, j != (i-1)%n)
    
    tour_edge_min_delta = np.full(n, np.inf)
    
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        best_delta = np.inf
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            c = tour[j]
            d = tour[(j + 1) % n]
            delta = edge_distance[a, c] + edge_distance[b, d] - edge_distance[a, b] - edge_distance[c, d]
            if delta < best_delta:
                best_delta = delta
        tour_edge_min_delta[i] = best_delta
    
    # Determine penalty for each tour edge based on how "easy" it is to break
    # Edges with small (or negative) min_delta are easy to break -> penalize more
    # Edges with large min_delta are hard to break -> penalize less
    
    finite_deltas = tour_edge_min_delta[np.isfinite(tour_edge_min_delta)]
    if len(finite_deltas) > 0:
        max_delta = np.max(finite_deltas)
        min_delta = np.min(finite_deltas)
        delta_range = max_delta - min_delta
        if delta_range < 1e-10:
            delta_range = 1.0
    else:
        max_delta = 1.0
        min_delta = 0.0
        delta_range = 1.0
    
    # Base penalty scale
    base_scale = avg_edge_dist * 0.4
    
    # Get usage statistics for diversification
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    usage_norm = edge_n_used / max_used
    
    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        d = tour_edge_min_delta[i]
        
        if np.isfinite(d):
            # Normalize delta: 0 means hardest to break, 1 means easiest
            if d <= 0:
                norm_d = 1.0  # Improving swap available, force break
            else:
                norm_d = 1.0 - (d - min_delta) / delta_range
                norm_d = max(0.0, min(1.0, norm_d))
        else:
            norm_d = 0.0
        
        # Penalty proportional to how easy it is to break
        p = base_scale * norm_d
        penalty[a, b] = p
        penalty[b, a] = p
    
    # Add mild diversification pressure from edge usage history
    penalty += base_scale * 0.15 * usage_norm
    
    # Ensure symmetry and zero diagonal
    penalty = (penalty + penalty.T) / 2.0
    np.fill_diagonal(penalty, 0.0)
    
    # Build output matrix
    D_out = edge_distance + penalty
    
    # Ensure finiteness
    D_out = np.nan_to_num(D_out, nan=0.0, posinf=avg_edge_dist * 10, neginf=0.0)
    
    # Ensure symmetry (numerical safety)
    D_out = (D_out + D_out.T) / 2.0
    np.fill_diagonal(D_out, 0.0)
    
    return D_out
