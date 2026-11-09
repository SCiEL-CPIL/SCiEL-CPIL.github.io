"""Combined CVRP heuristic: savings-based selection with capacity-awareness."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings-based heuristic with capacity-awareness.
    Returns 0 (depot) if no feasible node can be added or if savings are negative.
    
    Strategy:
    - When at depot, pick the nearest feasible node to start the route.
    - Otherwise, compute savings for each feasible node:
      saving(j) = dist(current, depot) + dist(depot, j) - dist(current, j)
    - If max saving is negative, close the route (go to depot).
    - Otherwise, pick the node with maximum saving, with a small capacity penalty
      to encourage efficient use of remaining capacity.
    """
    global _start_time, _DEADLINE
    
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
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Compute distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    # Calculate savings for each feasible node
    savings = d_current_depot + d_depot_j - d_current_j
    
    # If the maximum saving is negative, it is cheaper to close the route
    if np.max(savings) < 0:
        return int(depot)
    
    # Capacity-aware refinement:
    # Penalize nodes that would leave very little capacity (risk of stranding)
    # or nodes with very small demand (wasting capacity)
    demand_ratios = demands[feasible_nodes] / rest_capacity
    
    # Small penalty for extreme demand ratios
    # Ideal is around 0.3-0.7 of remaining capacity
    capacity_penalty = 0.05 * (demand_ratios - 0.5) ** 2
    
    # Adjust savings by capacity penalty (subtract penalty from savings)
    adjusted_savings = savings - capacity_penalty * d_current_j
    
    # Pick the node with the maximum adjusted saving
    max_idx = np.argmax(adjusted_savings)
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
