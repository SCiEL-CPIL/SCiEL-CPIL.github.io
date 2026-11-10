"""Improved heuristic for ACO-TSP: power-law distance + k-NN soft-min + consensus tour edges."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(dm, tour):
    """In-place 2-opt improvement on a closed tour list (node indices)."""
    n = len(tour)
    if n < 4:
        return tour
    best = sum(dm[tour[i], tour[(i + 1) % n]] for i in range(n))
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]
            for j in range(i + 1, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if delta < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    best += delta
                    improved = True
    return tour


def _nearest_neighbor_tour(dm, start=0):
    """Greedy nearest-neighbor tour from `start`."""
    n = dm.shape[0]
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = dm[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return tour


def _soft_min_scores(dm, k=8, temperature=0.15):
    """
    Smooth surrogate: for each node i, compute a soft-min over its k nearest
    neighbors' distances, producing a per-edge quality score.
    """
    n = dm.shape[0]
    eps = 1e-12
    
    dm_work = dm.copy()
    np.fill_diagonal(dm_work, np.inf)
    
    sorted_dm = np.sort(dm_work, axis=1)
    kth_dist = sorted_dm[:, k - 1]
    kth_dist = np.maximum(kth_dist, eps)
    
    scale_i = kth_dist[:, np.newaxis] * temperature
    scale_j = kth_dist[np.newaxis, :] * temperature
    scale = np.sqrt(scale_i * scale_j)
    scale = np.maximum(scale, eps)
    
    quality = np.exp(-dm / scale)
    np.fill_diagonal(quality, 0.0)
    
    return quality


def heuristics(distance_matrix):
    t0 = time.time()
    deadline = t0 + 280.0

    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    eps = 1e-9

    # Power-law base: use inverse squared distance for aggressive long-edge suppression
    # This creates a "negative curvature" effect in 2D geometry
    dm_safe = np.maximum(dm, eps)
    base_power = 1.0 / (dm_safe ** 2)
    np.fill_diagonal(base_power, 0.0)

    # k-NN soft-min component
    quality = _soft_min_scores(dm, k=8, temperature=0.15)

    # Combine base power-law with soft-min quality
    # Normalize both to mean 1 before combining
    base_power_mean = np.mean(base_power)
    if base_power_mean > eps:
        base_power_norm = base_power / base_power_mean
    else:
        base_power_norm = base_power

    quality_mean = np.mean(quality)
    if quality_mean > eps:
        quality_norm = quality / quality_mean
    else:
        quality_norm = quality

    # Weighted combination: emphasize power-law for long-edge suppression
    heuristic = 0.6 * base_power_norm + 0.4 * quality_norm

    # Consensus tour edge boost
    if time.time() < deadline:
        tour = _nearest_neighbor_tour(dm, 0)
        tour = _two_opt(dm, tour)
        
        tour_edge_boost = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            d_ab = dm[a, b]
            
            row_a = dm[a].copy()
            row_a[a] = np.inf
            row_a[b] = np.inf
            if n > 2:
                second_shortest_a = np.sort(row_a)[0]
            else:
                second_shortest_a = d_ab
            
            row_b = dm[b].copy()
            row_b[b] = np.inf
            row_b[a] = np.inf
            if n > 2:
                second_shortest_b = np.sort(row_b)[0]
            else:
                second_shortest_b = d_ab
            
            ref = max(second_shortest_a, second_shortest_b, d_ab)
            ratio = d_ab / max(ref, eps)
            tour_edge_boost[a, b] = ratio
            tour_edge_boost[b, a] = ratio
        
        heuristic = heuristic * (1.0 + 3.0 * tour_edge_boost)

    # Additional long-edge penalty using median
    upper_tri = dm[np.triu_indices(n, k=1)]
    med_len = np.median(upper_tri)
    long_penalty = np.exp(-np.maximum(0.0, dm / max(med_len, eps) - 1.0) * 0.5)
    heuristic = heuristic * long_penalty

    # Enforce positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so that the mean is 1
    mean_h = np.mean(heuristic)
    if mean_h > eps:
        heuristic = heuristic / mean_h

    return heuristic
# EVOLVE-BLOCK-END
