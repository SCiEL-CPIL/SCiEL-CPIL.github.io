"""Combined CVRP heuristic: savings-based selection with angular sweep and capacity awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None
_prev_node = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a hybrid savings + angular sweep heuristic.
    Returns 0 (depot) if no feasible node can be added, if savings are negative,
    or if the time limit is reached.
    
    Strategy:
    - When at depot, pick the nearest feasible node to start the route.
    - Otherwise, compute savings for each feasible node.
    - Add an angular penalty: penalize nodes that deviate from the current route direction.
    - If max adjusted saving is negative, close the route (go to depot).
    - Otherwise, pick the node with maximum adjusted saving.
    """
    global _start_time, _DEADLINE, _prev_node
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        _prev_node = current_node
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        _prev_node = current_node
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        _prev_node = current_node
        return int(feasible_nodes[min_idx])
    
    n_feasible = len(feasible_nodes)
    
    # Compute distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    # Calculate savings for each feasible node
    savings = d_current_depot + d_depot_j - d_current_j
    
    # Angular sweep: compute the angle of each node relative to the depot
    # and penalize nodes that deviate from the current route direction
    depot_coords = np.array([0.0, 0.0])  # We'll use relative angles from current to depot
    
    # Compute angle from depot to each feasible node
    # Using the distance matrix, we can approximate angles using the law of cosines
    # angle(j) = angle between vector(depot->current) and vector(depot->j)
    
    d_depot_current = distance_matrix[depot, current_node]
    
    # For each feasible node j, compute the angle at depot between current and j
    # Using law of cosines: cos(angle) = (d_depot_current^2 + d_depot_j^2 - d_current_j^2) / (2 * d_depot_current * d_depot_j)
    if d_depot_current > 1e-10:
        cos_angles = (d_depot_current**2 + d_depot_j**2 - d_current_j**2) / (2 * d_depot_current * d_depot_j)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)  # angles in [0, pi]
    else:
        angles = np.zeros(n_feasible)
    
    # Angular penalty: nodes that are far in angle from the current direction get penalized
    # This encourages the route to sweep around the depot in a consistent direction
    angular_weight = 0.5  # Weight for angular penalty
    angular_penalty = angular_weight * angles
    
    # Also add a small look-ahead penalty for nodes far from other feasible nodes
    if n_feasible > 1:
        dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        np.fill_diagonal(dist_matrix_feasible, np.inf)
        nearest_other_dist = np.min(dist_matrix_feasible, axis=1)
        look_ahead_penalty = 0.2 * nearest_other_dist
    else:
        look_ahead_penalty = np.zeros(1)
    
    # Adjusted savings: subtract angular and look-ahead penalties
    adjusted_savings = savings - angular_penalty - look_ahead_penalty
    
    # If the maximum adjusted saving is negative, it is cheaper to close the route
    if np.max(adjusted_savings) < 0:
        _prev_node = current_node
        return int(depot)
    
    # Pick the node with the maximum adjusted saving
    max_idx = np.argmax(adjusted_savings)
    _prev_node = current_node
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
