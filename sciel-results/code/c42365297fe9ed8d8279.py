"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dm, start=0):
    """Generate a tour using the nearest neighbor heuristic."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dm[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total


def _two_opt(tour, dm, max_iter=200):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing segment i+1..j
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
        if not improved:
            break
    return tour, best_len


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses an ensemble of reference tours to compute edge frequencies.
    Edges that appear frequently in multiple high-quality tours are given higher attractiveness.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # Initialize edge count matrix
    # Since the TSP is symmetric in distance but the tour direction matters for the "closed tour" 
    # pheromone update, we should consider if the heuristic should be symmetric or directed.
    # The problem states: "each ant adds 1/length to pheromone[i, j] for each step i to j of its closed tour, in travel direction only."
    # The heuristic is "edge-attractiveness". Usually in TSP ACO, eta[i,j] is symmetric if dist is symmetric.
    # However, if we count directed edges from the tours, we might get asymmetric heuristics if the tours are asymmetric.
    # But NN + 2-opt on symmetric distances usually produces tours where the reverse is equally valid.
    # Let's count directed edges (i, j) as they appear in the tour.
    # If a tour goes i->j, we increment count[i,j].
    
    edge_counts = np.zeros((n, n), dtype=float)
    
    # Generate multiple diverse reference tours
    # For n=50, we can afford several starts.
    # Let's try 10 different starting nodes.
    n_starts = 10
    
    for s in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        start_node = s % n
        # Generate NN tour from this start
        tour = _nearest_neighbor_tour(dm, start=start_node)
        
        # Improve with 2-opt
        # Limit iterations to keep it fast, as we have multiple starts
        tour, _ = _two_opt(tour, dm, max_iter=100)
        
        # Count edges in this tour
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_counts[u, v] += 1.0
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple inverse distance if we ran out of time
        base_heuristic = 1.0 / (dm + 1e-9)
        return np.maximum(base_heuristic, 1e-9)
    
    # Convert counts to heuristic values.
    # We want larger values to be more attractive.
    # The base heuristic (1/d) provides a scale.
    # If we just use counts, the magnitude might be small (e.g., 0 to 10).
    # Pheromone starts at 1.
    # Probability ~ P * eta.
    # If eta is small, it might be dominated by noise or initial pheromone uniformity?
    # Actually, if eta is uniform, it just scales all probabilities equally.
    # We need eta to differentiate edges.
    
    # Let's combine with a base inverse distance to ensure that very long edges are still penalized,
    # even if they appear in some tours (unlikely for good tours, but possible).
    # Or, simply use the counts directly, normalized.
    
    # Strategy: 
    # 1. Base heuristic: 1 / (dist + eps)
    # 2. Boost factor: 1 + edge_counts
    # This way, edges in the reference tours are boosted, but the relative attractiveness 
    # among non-reference edges is still governed by distance.
    
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Normalize counts to [0, 1] range for better scaling?
    # Max count is n_starts.
    # If an edge is in all 10 tours, count=10.
    # If in 0, count=0.
    
    # Let's try: heuristic = base_heuristic * (1 + edge_counts * boost_factor)
    # If boost_factor is 1, max boost is 11x.
    # This preserves the distance-based ranking for non-tour edges while strongly favoring tour edges.
    
    boost_factor = 1.0
    heuristic = base_heuristic * (1.0 + edge_counts * boost_factor)
    
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
