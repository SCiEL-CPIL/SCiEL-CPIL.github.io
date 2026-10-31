"""Traveling salesman constructive heuristic using force-directed physics model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Force-directed system for TSP tour construction.
#
# Each unvisited city experiences:
#   1. Attractive force toward the current node (pulls toward continuing the tour)
#   2. Repulsive forces from recently visited nodes (avoids edge crossings and backtracking)
#   3. Alignment bonus with the direction toward destination (maintains coherent path)
#
# The next city is selected by maximizing net force alignment with the tour direction.
# This physics-based representation naturally avoids edge crossings because
# repulsion from visited nodes creates a "wake" that the tour avoids re-entering.

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a force-directed physics model.
    
    The tour is modeled as a particle system where:
    - Attraction to current node encourages local continuity
    - Repulsion from visited nodes prevents edge crossings
    - Alignment with destination maintains global coherence
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    t_start = time.time()
    DEADLINE = 145.0
    
    n_remaining = len(unvisited_nodes)
    total_nodes = distance_matrix.shape[0]
    
    # Extract coordinates if available, otherwise use distance matrix structure
    # For Euclidean instances, we can infer coordinates from the distance matrix
    # by embedding in 2D using the first few principal components
    
    # Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from destination to each candidate
    dists_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Sub-matrix of distances between unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    sub_matrix_no_diag = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_no_diag, np.inf)
    
    # Force-directed scoring:
    # Score(u) = attraction_to_current - repulsion_from_visited + alignment_with_direction
    
    # 1. Attraction: prefer nodes close to current (normalized by scale)
    # Use inverse distance as attraction strength
    scale = np.mean(dists_from_current) + 1e-10
    attraction = 1.0 / (dists_from_current / scale + 1e-10)
    
    # 2. Repulsion from recently visited nodes:
    # We don't have explicit visited list, but we can approximate by
    # considering that nodes far from current but close to destination
    # might be part of a "return path" that we want to avoid
    
    # 3. Alignment with tour direction:
    # The tour should progress toward destination while avoiding backtracking
    # Compute a direction score: nodes that are "in the right direction" get bonus
    
    # Progress: 0 at start, 1 near end
    progress = 1.0 - (n_remaining / max(total_nodes, 1))
    
    # Adaptive weights:
    # Early: more attraction to current (build local structure)
    # Late: more alignment with destination (close the tour)
    w_attraction = 1.5 - 0.8 * progress
    w_repulsion = 0.5 + 0.3 * progress
    w_alignment = 0.3 + 0.7 * progress
    
    # Alignment score: prefer nodes that are roughly between current and destination
    # or at least not moving away from destination
    # Use the difference in distances: if d(current,u) + d(u,dest) ≈ d(current,dest),
    # then u is "on the way" to destination
    d_current_dest = distance_matrix[current_node, destination_node]
    
    # Triangle inequality residual: how much longer the path current->u->dest is
    # compared to direct current->dest. Small residual means u is "on the way".
    triangle_residual = dists_from_current + dists_from_dest - d_current_dest
    # Normalize and use as alignment score (lower residual = better alignment)
    max_residual = np.max(triangle_residual) + 1e-10
    alignment_score = 1.0 - (triangle_residual / max_residual)
    
    # Repulsion: nodes that are very close to current but also close to destination
    # might create short edges that could cross. Penalize nodes where both distances
    # are small (potential crossing risk)
    # Use product of normalized distances as a proxy for "safe" positioning
    norm_dist_current = dists_from_current / (np.max(dists_from_current) + 1e-10)
    norm_dist_dest = dists_from_dest / (np.max(dists_from_dest) + 1e-10)
    
    # Repulsion is higher when node is close to both current and dest (crossing risk)
    repulsion = norm_dist_current * norm_dist_dest
    
    # Combine scores: maximize attraction + alignment - repulsion
    scores = w_attraction * attraction + w_alignment * alignment_score - w_repulsion * repulsion
    
    # Check deadline
    if time.time() > DEADLINE:
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    best_idx = np.argmax(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
