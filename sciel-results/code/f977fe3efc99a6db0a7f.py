"""TSP constructive heuristic using angular deviation + savings-based scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0 - 5.0  # 5s safety margin

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using an angular-deviation heuristic that penalizes
    backtracking and edge crossings. For each candidate u, we score:
        score = d(current, u) + alpha * angular_penalty + beta * dest_proximity
    
    The angular penalty measures how much the direction from current->u deviates
    from the "ideal" straight-through direction (toward the centroid of remaining
    unvisited nodes or the destination). This discourages U-turns that create
    crossing edges in the final tour.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from unvisited to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For small remaining sets, destination proximity dominates
    if n_unvisited <= 3:
        total_cost = dist_from_current + 1.5 * dist_to_dest
        best_idx = int(np.argmin(total_cost))
        return int(unvisited_nodes[best_idx])
    
    # Compute the "ideal direction": we want to head roughly toward the 
    # centroid of unvisited nodes (or destination when close to end).
    # Since we only have a distance matrix (not coordinates), we approximate
    # angular deviation using the triangle inequality structure.
    #
    # Key insight: if going current -> u -> v is "straight", then
    # d(current, v) ≈ d(current, u) + d(u, v).
    # If it's a "turn-back", d(current, v) < d(current, u) + d(u, v) significantly.
    #
    # For each candidate u, find the nearest unvisited v (excluding u itself).
    # The "straightness" of the path current->u->v is measured by:
    #   straightness = d(current, v) / (d(current, u) + d(u, v))
    # Higher straightness (closer to 1) means less angular deviation.
    #
    # We penalize low straightness (high angular deviation).
    
    # Build sub-matrix for unvisited nodes
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(unvisited_dist_matrix, np.inf)
    
    # For each candidate u_i, find nearest other unvisited v_i
    nearest_dist = np.min(unvisited_dist_matrix, axis=1)
    nearest_idx = np.argmin(unvisited_dist_matrix, axis=1)
    
    # For each candidate u_i with nearest v_i:
    # d(current, v_i) from the full distance matrix
    v_nodes = unvisited_nodes[nearest_idx]
    d_current_to_v = distance_matrix[current_node, v_nodes]
    
    # Straightness ratio: 1.0 = perfectly straight, 0 = max deviation
    denom = dist_from_current + nearest_dist
    denom = np.maximum(denom, 1e-10)
    straightness = d_current_to_v / denom
    # Clamp to [0, 1]
    straightness = np.clip(straightness, 0.0, 1.0)
    
    # Angular penalty: higher when straightness is low (more turning)
    angular_penalty = 1.0 - straightness
    
    # Adaptive weight: stronger angular penalty when more nodes remain
    # (more opportunity for crossings), weaker when few remain
    alpha = 0.3 + 0.4 * (1.0 - n_unvisited / 50.0)
    
    # Destination proximity term: weight increases as we approach the end
    beta = 0.1 + 0.6 * (1.0 - n_unvisited / 50.0)
    
    # Total score: minimize this
    total_cost = dist_from_current + alpha * angular_penalty * np.max(dist_from_current) + beta * dist_to_dest
    
    best_idx = int(np.argmin(total_cost))
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
