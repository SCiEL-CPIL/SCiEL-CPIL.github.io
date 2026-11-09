import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a Greedy Savings Heuristic.
    Returns 0 (depot) if no feasible nodes or to close route.
    """
    if not unvisited_nodes:
        return depot

    # Filter nodes that fit in the remaining capacity
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if not feasible_nodes:
        return depot

    # Convert to numpy array for vectorization
    feasible_arr = np.array(feasible_nodes)
    
    # Calculate distances
    # d_cj: distance from current_node to each feasible node
    d_cj = distance_matrix[current_node, feasible_arr]
    # d_dj: distance from depot to each feasible node
    d_dj = distance_matrix[depot, feasible_arr]
    
    # Savings Heuristic Score:
    # We want to maximize savings s_ij = d_id + d_jd - d_ij
    # Since d_id is constant for the current step, maximizing s_ij is equivalent to
    # minimizing d_ij - d_jd
    scores = d_cj - d_dj
    
    # Find the node with the minimum score (maximum savings)
    min_idx = np.argmin(scores)
    best_node = feasible_arr[min_idx]
    
    return int(best_node)
