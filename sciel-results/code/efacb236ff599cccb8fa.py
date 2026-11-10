import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return depot

    # Identify feasible nodes
    # demands can be list/array or dict
    feasible = []
    for j in unvisited_nodes:
        # Check capacity
        if demands[j] <= rest_capacity:
            feasible.append(j)

    if not feasible:
        return depot

    # Convert to numpy array for vectorized distance calculation
    feasible_arr = np.array(feasible)
    
    # Distance from current node to each feasible node
    dist_from_current = distance_matrix[current_node, feasible_arr]
    
    # Distance from each feasible node to the depot
    dist_to_depot = distance_matrix[feasible_arr, depot]
    
    # Heuristic Score: Minimize the path length if we were to return to depot after visiting this node.
    # This balances proximity to current position with proximity to depot,
    # preventing the vehicle from drifting too far from the depot while capacity remains.
    scores = dist_from_current + dist_to_depot
    
    # Select the node with the minimum score
    min_idx = np.argmin(scores)
    best_node = feasible_arr[min_idx]
    
    return int(best_node)
