"""Heuristic matrix for ACO-TSP: Topological reinforcement via fast 2-opt NN tour."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 330s wall-clock limit, use 300s as safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0


def _two_opt(tour, dist):
    """Perform 2-opt improvement on a tour."""
    n = len(tour)
    improved = True
    # Limit iterations to avoid infinite loops in edge cases, though 2-opt always converges
    max_iter = 1000
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[(j+1) % n]
                # Check if reversing segment i..j reduces length
                # Current edges: (a,b) and (c,d)
                # New edges: (a,c) and (b,d)
                # Note: tour is a list of node indices.
                # dist is the distance matrix.
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d]:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour


def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Compute a fast Nearest Neighbor (NN) tour.
    2. Improve it with 2-opt to get a locally optimal tour.
    3. Construct a heuristic matrix where edges in this good tour are highly attractive,
       and other edges have a baseline geometric attractiveness (1/d).
    This 'primes' the ACO by making the ants likely to traverse known good edges early,
       allowing pheromone to reinforce a good solution quickly.
    """
    # Deadline guard: if we're too close to the limit, return a safe heuristic immediately.
    if time.time() > _DEADLINE - 10.0:
        d = np.asarray(distance_matrix, dtype=np.float64)
        return 1.0 / (d + 1e-9)

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # 1. Nearest Neighbor Tour
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        # Mask visited nodes with inf
        dists = d[current].copy()
        dists[visited] = np.inf
        # Ensure diagonal is inf if not already (it is 0, but visited[0] is True so it's masked)
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # 2. 2-Opt Improvement
    # We work with a list for easy reversal
    tour_list = list(tour)
    # Run 2-opt. For n=50, this is very fast.
    # We can run it a few times or just once. Once is usually sufficient for NN.
    tour_list = _two_opt(tour_list, d)
    
    # 3. Construct Heuristic Matrix
    # Base heuristic: 1 / d^2 (sharp local preference)
    # We use 1/d^2 as a baseline to maintain some geometric sense for non-tour edges
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    
    # Identify edges in the improved tour
    # Tour is a cycle: 0 -> ... -> 0
    # Edges are (tour[i], tour[(i+1)%n])
    heuristic = base_heuristic.copy()
    
    # Reinforce tour edges
    # We want these edges to be significantly more attractive.
    # Multiplying by a large factor ensures ants will likely pick them if available.
    # However, we must be careful not to make them so dominant that the colony gets stuck
    # if the 2-opt tour is suboptimal. But 2-opt on NN is usually quite good for n=50.
    # Let's use a multiplier of 100.
    for i in range(n):
        u = tour_list[i]
        v = tour_list[(i + 1) % n]
        heuristic[u, v] *= 100.0
        heuristic[v, u] *= 100.0
        
    # Ensure all values are finite and positive (defensive)
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0),
                         heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
