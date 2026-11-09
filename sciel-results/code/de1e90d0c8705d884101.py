"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a hybrid heuristic:
    - For small remaining sets (<=3), use exact lookahead
    - For larger sets, use a vectorized scoring heuristic balancing:
      * immediate distance from current node
      * distance to destination (for closing)
      * local connectivity penalty (min distance to other unvisited nodes)
    """
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    if time.time() - _start_time > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    unvisited = unvisited_nodes
    n_unvisited = len(unvisited)

    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited[0])

    # For 2 nodes: exact 2-step lookahead
    if n_unvisited == 2:
        i, j = unvisited
        cost_ij = distance_matrix[current_node, i] + distance_matrix[i, j] + distance_matrix[j, destination_node]
        cost_ji = distance_matrix[current_node, j] + distance_matrix[j, i] + distance_matrix[i, destination_node]
        return int(i if cost_ij <= cost_ji else j)

    # For 3 nodes: exact 3-step lookahead
    if n_unvisited == 3:
        best_node = unvisited[0]
        best_cost = float('inf')
        for first in unvisited:
            remaining = [x for x in unvisited if x != first]
            a, b = remaining[0], remaining[1]
            # Two possible orders for the remaining two
            cost1 = distance_matrix[current_node, first] + distance_matrix[first, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
            cost2 = distance_matrix[current_node, first] + distance_matrix[first, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
            cost = min(cost1, cost2)
            if cost < best_cost:
                best_cost = cost
                best_node = first
        return int(best_node)

    # For 4+ nodes: vectorized geometric heuristic
    # Distances from current node to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited]
    
    # Distance from current node to destination
    d_cur_dest = distance_matrix[current_node, destination_node]
    
    # Distances from all unvisited nodes to destination
    d_to_dest = distance_matrix[unvisited, destination_node]
    
    # Compute angle between (current -> candidate) and (current -> destination)
    # Using Law of Cosines: c^2 = a^2 + b^2 - 2ab cos(theta)
    # a = d_cur_dest, b = d_from_current, c = d_to_dest
    # cos(theta) = (a^2 + b^2 - c^2) / (2 * a * b)
    
    # Avoid division by zero
    denom = 2 * d_cur_dest * d_from_current
    # If d_cur_dest is 0 (current is destination), angle is undefined, fall back to nearest neighbor
    if d_cur_dest < 1e-9:
         return int(unvisited[np.argmin(d_from_current)])
         
    # Mask where denom is 0
    safe_denom = np.where(denom > 1e-9, denom, 1.0)
    cos_theta = (d_cur_dest**2 + d_from_current**2 - d_to_dest**2) / safe_denom
    
    # Clip cos_theta to [-1, 1] to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle deviation score: 1 - cos(theta) is 0 when aligned, 2 when opposite
    # We want to minimize this, so we minimize (1 - cos_theta)
    angle_score = 1.0 - cos_theta
    
    # Adaptive weights: 
    # Early in tour (many unvisited), we might want to stay closer to nearest neighbor to cover area?
    # Or, the directive says "tracing the convex hull... before filling interior".
    # Minimizing angle towards destination tends to keep the tour "tight" around the destination.
    # Let's use a blend: Distance + Angle.
    
    if n_unvisited <= 10:
        w_dist = 0.6
        w_angle = 0.4
    elif n_unvisited <= 20:
        w_dist = 0.7
        w_angle = 0.3
    else:
        w_dist = 0.8
        w_angle = 0.2

    # Normalize components
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-9)
    angle_score_norm = angle_score / (np.max(angle_score) + 1e-9)
    
    scores = w_dist * d_from_current_norm + w_angle * angle_score_norm

    best_idx = np.argmin(scores)
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
