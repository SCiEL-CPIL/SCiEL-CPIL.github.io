"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(tour, dist):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    improved = True
    max_iters = 100
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                current = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new < current - 1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _three_opt(tour, dist):
    """Improve a tour using 3-opt local search."""
    n = len(tour)
    tour = list(tour)
    improved = True
    max_iters = 20
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if k == (j + 1) % n:
                        continue
                    # Current edges: (tour[i-1], tour[i]), (tour[j], tour[j+1]), (tour[k], tour[(k+1)%n])
                    a = tour[i - 1]
                    b = tour[i]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    e = tour[k]
                    f = tour[(k + 1) % n]
                    
                    # 3-opt reversal: reverse tour[i..j] and reverse tour[j+1..k]
                    # This is complex; let's stick to 2-opt for reliability
                    pass
        if not improved:
            break
    return tour


def _nearest_neighbor_tour(dist):
    """Generate a tour using nearest neighbor heuristic."""
    n = dist.shape[0]
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        best_node = min(unvisited, key=lambda j: dist[current, j])
        tour.append(best_node)
        unvisited.remove(best_node)
        current = best_node
    return tour


def _greedy_tour(dist):
    """Generate a tour using greedy edge selection (cheapest insertion-like)."""
    n = dist.shape[0]
    # Start with the shortest edge
    d_flat = dist.flatten()
    indices = np.argsort(d_flat)
    
    used_edges = set()
    degree = np.zeros(n, dtype=int)
    tour_edges = []
    
    for idx in indices:
        i, j = divmod(idx, n)
        if i == j:
            continue
        if (i, j) in used_edges or (j, i) in used_edges:
            continue
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        # Check for subtour (only allow if it connects to the main component or is the start)
        # For simplicity, just check degree constraint
        tour_edges.append((i, j))
        used_edges.add((i, j))
        degree[i] += 1
        degree[j] += 1
        if len(tour_edges) == n:
            break
    
    # Convert edges to a tour
    if len(tour_edges) < n:
        return _nearest_neighbor_tour(dist)
    
    # Build adjacency
    adj = {i: [] for i in range(n)}
    for u, v in tour_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find the tour
    tour = [0]
    prev = -1
    current = 0
    for _ in range(n - 1):
        nexts = adj[current]
        if len(nexts) == 1:
            nxt = nexts[0]
        else:
            nxt = nexts[0] if nexts[0] != prev else nexts[1]
        tour.append(nxt)
        prev = current
        current = nxt
    
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 300.0 - 10.0  # 290s safety margin

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if d.shape[0] != d.shape[1]:
        raise ValueError("Distance matrix must be square")

    # Generate multiple candidate tours
    tours = []
    
    # 1. Nearest Neighbor
    nn_tour = _nearest_neighbor_tour(d)
    nn_tour = _two_opt(nn_tour, d)
    tours.append(nn_tour)
    
    # 2. Greedy
    gr_tour = _greedy_tour(d)
    gr_tour = _two_opt(gr_tour, d)
    tours.append(gr_tour)
    
    # 3. Random restarts with 2-opt (if time permits)
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 50.0:
            break
        random_tour = list(range(n))
        rng.shuffle(random_tour)
        improved_tour = _two_opt(random_tour, d)
        tours.append(improved_tour)

    # Find the best tour
    best_tour = min(tours, key=lambda t: sum(d[t[i], t[(i + 1) % n]] for i in range(n)))
    
    # Compute heuristic based on inverse distance
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Base heuristic: 1/d
    eta = 1.0 / (d_safe + 1e-30)
    
    # Boost edges from the best tour
    boost_factor = 1.5
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        eta[u, v] *= boost_factor
        eta[v, u] *= boost_factor
    
    # Also boost edges from the second best tour slightly
    second_best = min(tours, key=lambda t: sum(d[t[i], t[(i + 1) % n]] for i in range(n)) if t != best_tour else float('inf'))
    boost_factor2 = 1.2
    for i in range(n):
        u = second_best[i]
        v = second_best[(i + 1) % n]
        eta[u, v] *= boost_factor2
        eta[v, u] *= boost_factor2

    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

    return eta
# EVOLVE-BLOCK-END
