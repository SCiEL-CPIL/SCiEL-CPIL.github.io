import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Convert to numpy arrays if necessary for vectorized operations
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter for feasible nodes based on capacity
    # demands is likely an array or list indexed by node id
    # rest_capacity is the remaining capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited[feasible_mask]
    
    # Calculate scores for feasible nodes
    # Score = dist(current, node) - dist(depot, node)
    # This maximizes the "savings" of not returning to depot yet
    
    dist_current = distance_matrix[current_node, feasible_nodes]
    dist_depot = distance_matrix[depot, feasible_nodes]
    
    scores = dist_current - dist_depot
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
