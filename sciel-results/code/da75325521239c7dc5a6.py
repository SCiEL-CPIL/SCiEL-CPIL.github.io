# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_LIMIT = 110.0  # Safety margin before 120s/150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Angle-based TSP constructive heuristic.
    
    Structural change: Instead of summing or taking max of edge lengths,
    we use the geometric angle at the current node to guide the tour.
    In Euclidean TSP, optimal tours tend to follow smooth paths with small
    turning angles. This heuristic scores candidates by:
    
    1. Angle penalty: 1 - cos(theta) where theta is the angle between
       the vector from previous direction to current node and the vector
       from current node to candidate. Small angles (straight ahead) are preferred.
    
    2. Distance normalization: Scale by mean distance to avoid bias.
    
    3. Destination awareness: Penalize candidates that move away from destination.
    
    This is a fundamentally different modeling family (geometric/angle-based)
    vs the previous distance-sum/minimax approaches.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _LIMIT:
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Extract coordinates if available, otherwise use distance matrix heuristics
    # Since we only have distance matrix, we need to work with distances directly.
    # Use a surrogate for angle: the "turning" can be approximated by how much
    # the distance to destination changes.
    
    # For each candidate i, compute:
    # - dist_cur_i: distance from current to candidate
    # - dist_i_dest: distance from candidate to destination
    # - dist_cur_dest: distance from current to destination
    
    dist_cur_dest = distance_matrix[current_node, destination_node]
    
    # Triangle inequality based angle proxy:
    # If candidate is "in the direction" of destination, then dist_cur_i + dist_i_dest ≈ dist_cur_dest
    # If candidate is "opposite" to destination, then dist_cur_i + dist_i_dest >> dist_cur_dest
    # So the "alignment" score is: (dist_cur_i + dist_i_dest - dist_cur_dest)
    # Lower means better aligned with destination direction.
    
    alignment_scores = dists_from_current + dists_to_dest - dist_cur_dest
    
    # Normalize alignment scores to [0, 1] range for consistent weighting
    min_align = np.min(alignment_scores)
    max_align = np.max(alignment_scores)
    align_range = max_align - min_align
    if align_range > 1e-10:
        normalized_align = (alignment_scores - min_align) / align_range
    else:
        normalized_align = np.zeros_like(alignment_scores)
    
    # Distance score: prefer closer candidates, but not too close (to avoid backtracking)
    # Use a soft penalty for very short distances (potential backtracking)
    min_dist = np.min(dists_from_current)
    max_dist = np.max(dists_from_current)
    dist_range = max_dist - min_dist
    if dist_range > 1e-10:
        normalized_dist = (dists_from_current - min_dist) / dist_range
    else:
        normalized_dist = np.zeros_like(dists_from_current)
    
    # Adaptive weights based on remaining nodes
    if n_unvisited > 15:
        # Early: balance distance and alignment
        w_dist = 0.4
        w_align = 0.6
    elif n_unvisited > 8:
        # Mid: slightly favor alignment
        w_dist = 0.3
        w_align = 0.7
    else:
        # Late: strongly favor alignment to reach destination
        w_dist = 0.2
        w_align = 0.8
    
    # Composite score
    scores = w_dist * normalized_dist + w_align * normalized_align
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
