"""Guided local search penalty for TSP; combine tour-specific GLS with multi-strategy search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0
_start_time = None


def _two_opt(D, tour, max_passes=10):
    """2-opt on distance matrix D starting from tour. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return list(tour)
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
    
    n = edge_distance.shape[0]
    if n <= 1:
        return edge_distance.copy()
    
    tour = list(local_opt_tour)
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple penalty to be safe
        max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
        return edge_distance * (1.0 + 0.1 * edge_n_used / (1.0 + max_used))
    
    # Compute median distance for scaling
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0
    
    # Build set of edges in current tour for quick lookup
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u != v:
            tour_edges.add((min(u, v), max(u, v)))
    
    # Try multiple penalty strategies
    # Each strategy: (base_mult, decay, use_global, description)
    # - base_mult: multiplier for penalty
    # - decay: power of usage (1.0 = linear, 2.0 = squared)
    # - use_global: if True, penalize all used edges; if False, only tour edges
    
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 1
    if max_used < 1e-10:
        max_used = 1.0
    
    strategies = [
        # Tour-specific penalties (original approach refined)
        (0.05, 1.0, False, "mild tour linear"),
        (0.10, 1.0, False, "moderate tour linear"),
        (0.15, 1.0, False, "strong tour linear"),
        (0.05, 1.5, False, "mild tour 1.5-power"),
        (0.10, 1.5, False, "moderate tour 1.5-power"),
        (0.03, 2.0, False, "mild tour squared"),
        (0.06, 2.0, False, "moderate tour squared"),
        # Global penalties (inspiration approach)
        (0.05, 1.0, True, "mild global linear"),
        (0.10, 1.0, True, "moderate global linear"),
        (0.03, 1.5, True, "mild global 1.5-power"),
        (0.06, 1.5, True, "moderate global 1.5-power"),
        (0.02, 2.0, True, "mild global squared"),
    ]
    
    best_D_pen = None
    best_len = float('inf')
    
    for base_mult, decay, use_global, desc in strategies:
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        D_pen = edge_distance.copy()
        
        if use_global:
            # Penalize all edges based on usage
            usage = edge_n_used / max_used
            penalty = base_mult * med_dist * (usage ** decay)
            D_pen += penalty
        else:
            # Penalize only edges in the current tour
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                if u == v:
                    continue
                count = edge_n_used[u, v]
                if count > 0:
                    # Scale by usage relative to max, raised to decay power
                    usage_ratio = count / max_used
                    penalty = base_mult * med_dist * (usage_ratio ** decay)
                    D_pen[u, v] += penalty
                    D_pen[v, u] += penalty
        
        # Ensure symmetry and zero diagonal
        D_pen = (D_pen + D_pen.T) / 2.0
        np.fill_diagonal(D_pen, 0.0)
        
        # Run 2-opt on the penalized matrix
        tour_pen = _two_opt(D_pen, tour, max_passes=10)
        
        # Evaluate on true distances
        true_len = _closed_len(edge_distance, tour_pen)
        
        if true_len < best_len:
            best_len = true_len
            best_D_pen = D_pen
    
    # If we found a better tour, return the corresponding penalized matrix
    if best_D_pen is not None:
        return best_D_pen
    else:
        # Fallback: return original distance
        return edge_distance.copy()
# EVOLVE-BLOCK-END
