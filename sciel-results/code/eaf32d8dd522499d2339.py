"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(dist, tour, max_passes=10):
    """Run 2-opt improvement on a tour given a distance matrix."""
    n = len(tour)
    tour = list(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # Check if reversing segment i+1..j improves
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
        if not improved:
            break
    return tour


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Applies penalties to edges based on their usage count, with a smooth
    decay and adaptive scaling. The penalty is designed to be strong enough
    to break local optima but not so strong as to completely distort the
    distance landscape.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = edge_distance.shape[0]
    
    # Get edges in the current tour
    tour = list(local_opt_tour)
    tour_edges = set()
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        if u > v:
            u, v = v, u
        tour_edges.add((u, v))
    
    # Compute penalty for each edge based on usage
    # Use a smooth penalty function that grows with usage but saturates
    max_used = edge_n_used.max()
    if max_used == 0:
        max_used = 1.0
    
    # Normalize usage counts
    normalized_usage = edge_n_used / max_used
    
    # Apply penalty: edges in the current tour get a stronger penalty
    # The penalty is multiplicative and smooth
    # Base penalty rate: scales with normalized usage
    # Additional boost for edges in the current tour
    penalty = np.zeros_like(edge_n_used)
    for (u, v) in tour_edges:
        # Penalty for edges in the current tour
        # Use a smooth function: 1 + alpha * usage / (1 + usage)
        # This gives a penalty that is bounded but grows with usage
        usage_val = edge_n_used[u, v]
        penalty[u, v] = 0.15 * usage_val / (1.0 + usage_val)
        penalty[v, u] = penalty[u, v]
    
    # Add a small global penalty for all frequently used edges
    # This helps escape from the basin even for edges not in the current tour
    global_penalty = 0.05 * normalized_usage / (1.0 + normalized_usage)
    
    # Combine penalties
    total_penalty = penalty + global_penalty
    
    # Create the perturbed distance matrix
    # Multiply by (1 + penalty) to increase the cost of penalized edges
    perturbed = edge_distance * (1.0 + total_penalty)
    
    # Ensure symmetry and finite values
    perturbed = (perturbed + perturbed.T) / 2.0
    np.fill_diagonal(perturbed, 0.0)
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
